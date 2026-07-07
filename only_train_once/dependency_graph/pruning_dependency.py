import os           # 这里导入 os 模块，主要用于获取当前文件路径、目录操作等
import sys          # 这里导入 sys 模块，主要用于在运行时向系统环境添加路径，方便直接从父级目录导入模块
import pprint

currentdir = os.path.dirname(os.path.realpath(__file__))    # 获取当前文件所在目录
parentdir = os.path.dirname(currentdir)                     # 获取父级目录
sys.path.append(parentdir)                                  # 将父级目录添加到 Python 搜索路径，确保可以导入该目录下的模块

from graph.node_group import NodeGroup                      # 从自定义模块 graph.node_group 中导入 NodeGroup 类，用于管理节点的组合（组件）
from operation.operator import UNPRUNABLE_BASIC_OPERATORS, UNPRUNABLE_COMPOSED_OPERATORS        # 从自定义模块 operation.operator 中导入不可剪枝的基础算子和复合算子集合
from transform import is_spread_transformation, TensorTransform, SPREAD_TRANSFORM_MAP           # 从自定义模块 transform 中导入相关的变换检测和处理方法


def get_non_stem_nodes(graph, skip_node_ids=set()):
    """
    获取图中非“stem”节点的列表，过滤掉 skip_node_ids 中的节点。
    stem 节点通常是输入/输出或关键节点等，这些可能不需要参与剪枝。
    同时，这里还排除了 is_concat(axis=1) 和 is_dummy() 节点。
    判断主干节点具体逻辑如下：
    1. 检查节点是否有操作对象 (self.op)：
        如果节点没有关联的操作对象 (self.op 为 None)，则该节点不是主干节点。
    2. 如果节点有关联的操作对象：
        检查操作是否为“基本操作” (self.op.is_basic)：
        如果操作是基本操作，则该节点是否为主干节点取决于其操作对象自身的 is_stem 属性 (self.op.is_stem)。在 BasicOperator 类中，
        is_stem 属性默认为 False，但可以在具体算子初始化时设置。
    3. 如果操作不是“基本操作”：
        则进一步判断该节点的操作类型。如果节点是以下任何一种类型，它被视为主干节点：
        卷积操作 (self.is_conv())：操作名称为 "Conv" 或 "conv"。
        转置卷积操作 (self.is_convtranspose())：操作名称为 "ConvTranspose" 或 "convtranspose"。
        线性（全连接）或 Gemm 操作 (self.is_linear())：操作名称为 "Linear"、"linear"、"Gemm" 或 "gemm"。
    总结来说，一个节点被判断为“主干节点”的条件是：
    它有一个关联的操作对象。并且，要么它的操作对象是“基本操作”且其 is_stem 标记为 True。
    要么它的操作对象不是“基本操作”，但其操作类型是卷积、转置卷积或线性层。

    论文定义的节点定义：
    主干顶点（Stem Vertices）定义：包含可训练参数，能够改变输入张量的形状（如卷积层 Conv、全连接层 Linear）。
    辅助顶点（Accessory Vertices）定义：单输入单输出，可能含可训练参数（如批归一化 BatchNorm、激活函数 ReLU）。
    连接顶点：形状依赖型（Shape-Dependent, SD）：输入需保持相同形状（如 Add）。形状独立型（Shape-Independent, SID）：输入形状可不同（如 Concat 沿通道维度拼接卷积层的输出）。
    未知顶点（Unknown Vertices）定义：未明确类型或操作不确定的顶点。
    """
    non_stem_nodes = list()     # 非“stem”节点的列表
    for node in graph.nodes.values():
        # 遍历图中的每个节点
        if node.id in skip_node_ids:
            # 跳过 skip_node_ids 中的节点
            continue
        # 如果节点既不是 stem，也不是 concat(axis=1)，且不是 dummy，则加入非 stem 列表
        # node.is_concat(axis=1)：判断该节点是否为在轴 1 上执行拼接（concatenate）操作的节点。
        # node.is_dummy()：判断该节点是否为“dummy”节点，例如 dummy_input、dummy_output 等。
        if not node.is_stem() and not node.is_concat(axis=1) and not node.is_dummy():
            non_stem_nodes.append(node)
    return non_stem_nodes

def get_non_stem_node_groups(graph, nodes):
    """
    基于给定的非 stem 节点列表 `nodes`，从计算图中找出连通分量（Connected Components），
    并将每个连通分量表示为一个 NodeGroup 实例。
    这个函数通过深度优先搜索（DFS）来识别互相可达的节点集合，从而构建节点组。
    参数:
        graph (Graph): 表示神经网络结构的图对象。
        nodes (list[Node]): 非 stem 节点列表，这些节点已经被筛选出来用于分析。
    返回:
        list[NodeGroup]: 一组 NodeGroup 对象，每个对象对应图中的一个连通分量。
    """
    # 初始化一个空列表，用于存储最终形成的节点组（即各个连通分量）
    node_groups = []
    # 创建一个字典 `visited` 来记录所有输入节点的访问状态，初始为未访问（False）
    visited = dict()
    for node in nodes:
        visited[node.id] = False  # 所有输入节点初始化为未访问
    def dfs_helper(graph, node, cc):
        """
        深度优先搜索（DFS）辅助函数，用于遍历当前节点及其连接的前后节点，并将其加入当前连通分量（cc）中。
        参数:
            graph (Graph): 当前处理的图对象。
            node (Node): 当前正在处理的节点。
            cc (NodeGroup): 当前构建的连通分量节点组。
        """
        visited[node.id] = True   # 标记当前节点为已访问
        cc.add_node(node)         # 将当前节点加入当前连通分量中

        for node_out in graph.outgoing(node):
            # 遍历当前节点的所有后继节点（outgoing edges）
            if node_out.id in visited:
                # 如果该节点在当前待访问的 nodes 列表中且尚未被访问
                if not visited[node_out.id]:
                    # 节点未被访问
                    dfs_helper(graph, node_out, cc)  # 继续递归 DFS

        for node_in in graph.incoming(node):
            # 遍历当前节点的所有前驱节点（incoming edges）
            if node_in.id in visited:
                # 如果该节点在当前待访问的 nodes 列表中且尚未被访问
                if not visited[node_in.id]:
                    # 节点未被访问
                    dfs_helper(graph, node_in, cc)  # 继续递归 DFS

    # 主循环：对所有未访问的节点执行 DFS，寻找连通分量
    for node in nodes:
        # 遍历所有非 stem 节点
        if not visited[node.id]:  
            # 只有未访问过的节点才开始一次新的 DFS
            connected_component = NodeGroup()               # 创建一个新的节点组实例
            dfs_helper(graph, node, connected_component)    # 启动 DFS 构建连通分量
            node_groups.append(connected_component)         # 将找到的连通分量添加到结果列表中

    return node_groups  # 返回最终构建好的所有节点组（连通分量）

def grow_non_stem_node_groups(graph, node_groups, skip_node_ids=set()):
    """
    对每个非 stem 节点组进行扩展操作。
    这个函数迭代 `node_groups` 列表中的每一个 `NodeGroup` 对象。
    对于每个 `NodeGroup`，它调用 `grow_non_stem_node_group` 函数，
    目的是将与当前非 stem 节点组紧密相关的 "stem" 节点或其他特定类型的节点（例如，执行通道拼接操作的 `concat(axis=1)` 节点）也包含到该节点组中。
    扩展的逻辑通常是从当前组内的节点出发，沿着图的边（主要是入边，即向前追溯）进行搜索，直到遇到符合特定条件的节点（如 stem 节点、concat 节点）或者
    遇到在 `skip_node_ids` 集合中指定的应被跳过的节点。
    这个过程有助于形成更完整的、在剪枝时应被视为一个单元的逻辑块。
    例如，一个非 stem 的激活层可能需要将其前面的 stem 卷积层一起考虑。
    参数:
        graph (Graph): 表示整个计算图的 Graph 对象。
        node_groups (list[NodeGroup]): 一个列表，包含了之前步骤中识别出的非 stem 节点组。这些是需要被扩展的初始节点组。
        skip_node_ids (set, optional): 一个包含节点 ID 的集合。在扩展过程中，如果遇到 ID 在此集合中的节点，则应停止沿该路径的扩展。默认为空集合。
    返回:
        list[NodeGroup]: 经过扩展操作后的节点组列表。
                         注意：这个函数是原地修改 `node_groups` 列表中的 `NodeGroup` 对象的，
                         所以返回的列表与传入的列表是同一个对象，但其内部的 `NodeGroup` 可能已经改变。
    """
    # 遍历传入的每一个非 stem 节点组
    for node_group in node_groups:
        # 对当前的 node_group 调用辅助函数 grow_non_stem_node_group 进行扩展。
        # graph 对象提供了图的结构信息。
        # node_group 是当前要被扩展的组。
        # skip_node_ids 用于控制扩展的边界。
        grow_non_stem_node_group(graph, node_group, skip_node_ids)
    # 返回修改后的 node_groups 列表。
    # 由于 NodeGroup 对象是可变的，并且 grow_non_stem_node_group 是原地修改 node_group，所以这里返回的 node_groups 列表中的元素是已经被扩展过的。
    return node_groups

def grow_non_stem_node_group(graph, node_group, skip_node_ids=set()):
    """
    对单个节点组执行扩展操作。
    从给定组中的节点开始，向前（incoming）DFS 寻找可能需要包含进该组的节点（例如 stem 或 concat）。
    """
    visited = {}    # 创建一个字典，用于记录节点的访问状态
    for node_id in graph.nodes:
        # 遍历图中的节点
        visited[node_id] = False if node_id not in skip_node_ids else True  # 如果在 跳过列表 中视为已访问，以免重复搜索

    new_nodes = list()  # 新发现的节点列表
    def dfs_helper(graph, node):
        # 当遇到 stem 或 concat(axis=1) 且节点不是 dummy（dummy_input, dummy_output） 时，将其加入 new_nodes 后结束搜索
        if (node.is_stem() or node.is_concat(axis=1)) and not node.is_dummy():
            new_nodes.append(node)
            return 
        visited[node.id] = True # 标记当前节点为已访问
        for node_in in graph.incoming(node):
            # 遍历当前节点的所有前驱节点
            if not visited[node_in.id]:
                # 如果前驱节点未被访问过
                dfs_helper(graph, node_in)  # 递归调用，继续向上查找

    # 对 group 内部所有节点进行 DFS，查找可扩展到的节点
    for node in node_group:
        if not visited[node.id]:    # 如果当前节点未被访问过
            dfs_helper(graph, node) # 递归调用，继续向下查找

    # 将新查找到的节点添加到 node_group 中
    node_group.add_nodes(new_nodes)

def merge_node_groups(node_groups):
    """
    如果多个节点组间有公共节点，则将它们合并。
    在合并后会不断检查新的组是否还可与其他组合并，直至无法再合并为止。
    参数:
        node_groups (list[NodeGroup]): 一个包含多个节点组的列表，每个节点组是 NodeGroup 类型的实例。
    返回:
        list[NodeGroup]: 合并后的节点组列表。
    """
    # 使用集合存储待合并的节点组，方便快速查找和删除
    pool = set(node_groups)  # 创建一个集合，用于存储待合并的节点组
    merged_node_groups = []  # 创建一个列表，用于存储已合并的节点组

    # 当还有未处理的节点组时，继续合并操作
    while pool: 
        # 从 pool 中取出一个节点组，作为当前正在处理的节点组
        merged_node_groups.append(pool.pop())  # 从 pool 中取出一个节点组，添加到 merged_node_groups 中

        # 不断尝试将当前节点组与 pool 中的其他节点组合并
        while True:
            # 遍历 pool 中剩余的节点组
            for cc in pool:  
                # 检查当前节点组（merged_node_groups[-1]）与 pool 中的节点组（cc）是否有公共节点
                # 使用 `nodes.keys()` 获取节点组中所有节点的键，并检查它们是否有交集
                if merged_node_groups[-1].nodes.keys() & cc.nodes.keys():
                    # 如果有交集，则将 cc 合并到当前节点组中
                    merged_node_groups[-1].merge(cc)
                    # 从 pool 中移除已合并的节点组 cc
                    pool.remove(cc)
                    # 由于 pool 被修改，跳出当前循环，重新开始检查
                    break
            else:
                # 如果没有找到可以合并的节点组，结束当前合并循环
                break

    # 返回合并后的节点组列表
    return merged_node_groups

def get_remaining_nodes(node_groups, all_nodes):
    """
    获取未被任何 node_group 包含的剩余节点列表（排除 dummy_input 和 dummy_output）。
    如果节点属于 NodeGroupComposedOp，还需特殊处理排除输出节点自身。
    参数:
        node_groups (list[NodeGroup]): 已经分组的节点组列表。
        all_nodes (dict): 图中所有节点的字典，键为节点 ID，值为节点对象。
    返回:
        list[Node]: 未被任何节点组包含的剩余节点列表。
    """
    remaining_nodes = []    # 用于存储未被包含的节点
    included_nodes = []     # 用于存储已被包含的节点 ID

    # 遍历所有节点组，收集已被包含的节点 ID
    for node_group in node_groups:
        if type(node_group).__name__ == 'NodeGroupComposedOp':
            # 如果节点组是复合操作节点组（NodeGroupComposedOp），
            # 从一个节点组 (node_group) 中提取所有非输出节点的 ID，并将它们添加到 included_nodes 列表中。
            included_nodes.extend([
                node.id for node in node_group.nodes.values()
                if node.id not in node_group.output_nodes
            ])
        else:
            # 如果是普通节点组，收集所有节点的 ID。
            included_nodes.extend([
                node.id for node in node_group.nodes.values()
            ])

    # 遍历所有节点，找出未被包含的节点
    remaining_nodes = [
        all_nodes[node_id]                                              # 获取节点对象
        for node_id in all_nodes                                        # 遍历所有节点 ID
        if node_id not in included_nodes                                # 如果节点 ID 不在已包含的节点列表中
        and (node_id != 'dummy_input' and node_id != 'dummy_output')    # 排除 dummy_input 和 dummy_output
    ]

    return remaining_nodes  # 返回剩余节点列表
    
def group_individual_nodes(individual_nodes):
    """
    将剩余的单节点包装成单独的 NodeGroup（即每个 NodeGroup 只有一个节点）。
    参数:
        individual_nodes (list[Node]): 一个包含独立节点的列表，这些节点未被分配到任何节点组中。
    返回:
        list[NodeGroup]: 一个列表，其中每个 NodeGroup 只包含一个节点。
    """
    singleton_node_groups = list()  # 初始化一个空列表，用于存储单节点的 NodeGroup

    # 遍历每个独立节点
    for node in individual_nodes:
        node_group = NodeGroup()                    # 创建一个新的 NodeGroup 实例
        node_group.add_node(node)                   # 将当前节点添加到这个新的 NodeGroup 中
        singleton_node_groups.append(node_group)    # 将这个单节点的 NodeGroup 添加到结果列表中
    return singleton_node_groups                    # 返回包含所有单节点 NodeGroup 的列表
def group_nodes_composed_operator(graph):
    """
    在已有复合操作节点组 graph.node_groups 的基础上进行扩展，具体表现为：
    1. 基于现有节点组进行深度优先搜索（DFS）：从当前节点组中的每个节点出发，反向查找所有可以到达的、与该节点组相关的其他节点。
    2. 将新发现的相关节点加入当前节点组：通过扩展原有节点组的内容，使复合操作结构更加完整（如识别出更多属于同一复合操作的子节点）。
    3. 更新节点组属性：包括设置等价性（equivalent）、输出节点（output_node）等信息，确保合并后的节点组在后续剪枝分析中能正确表示一个完整的复合操作逻辑。
    参数:
        graph (Graph): 表示神经网络结构的图对象，包含节点、边及节点组等信息。
    返回:
        list[NodeGroup]: 一个列表，包含扩展后的复合操作节点组。
    """
    # adj_nodes: 一个集合，用于在DFS过程中累积找到的与当前处理的节点组相关的节点。
    # 它被声明为 nonlocal，以便在嵌套的 dfs_helper 函数中可以被修改。
    adj_nodes = set()
    
    # dfs_helper: 一个深度优先搜索（DFS）的辅助函数。
    # 它从 cur_node 开始，沿着图的入边（incoming edges）进行反向DFS。
    # 目的是找到一条从 cur_node 到 dst_ids 中任何一个节点的路径。
    # 如果找到了这样的路径，路径上的所有节点都会被添加到 nonlocal 的 adj_nodes 集合中。
    def dfs_helper(graph, cur_node, dst_ids, visited, path, verbose=False):
        # nonlocal adj_nodes 声明允许内部函数修改外部函数作用域中的 adj_nodes变量。
        nonlocal adj_nodes
        # 将当前节点标记为已访问。
        visited[cur_node.id] = True
        # 将当前节点添加到当前搜索路径中。
        path.append(cur_node)
        # 检查当前节点是否是目标节点集 (dst_ids) 中的一员。
        if cur_node.id in dst_ids:
            # 如果是目标节点，意味着找到了一条从起始节点到某个目标节点的路径。
            # 将这条路径上的所有节点都加入到 adj_nodes 集合中。
            adj_nodes = adj_nodes.union(set(path))
            # 返回 True 表示找到了目标。
            return True
        # 遍历当前节点的所有入边连接的节点 (node_in)。
        for node_in in graph.incoming(cur_node):
            # 如果入边节点尚未被访问过。
            if not visited[node_in.id]:
                # 递归调用 dfs_helper，从该入边节点继续搜索。
                # 注意：这里如果 dfs_helper 返回 True，当前实现并没有立即停止其他分支的搜索，
                # 而是会继续探索其他路径，这可能导致 adj_nodes 包含多条路径上的节点。
                dfs_helper(graph, node_in, dst_ids, visited, path, verbose)
        # 当从当前节点出发的所有路径都探索完毕后，将当前节点从路径中移除（回溯）。
        path.pop()
        # 如果 adj_nodes 不为空（即至少找到了一条路径），则返回 True，否则返回 False。
        return False if len(adj_nodes) == 0 else True
    
    # node_groups: 一个列表，用于存储经过处理和扩展后的节点组。
    node_groups = list()

    # old_node_group_ids: 一个列表，用于存储原始 graph.node_groups 中的所有节点组ID，以便后续将它们从 graph.node_groups 中删除。
    old_node_group_ids = list()

    # 遍历 graph.node_groups 字典中的每一个节点组ID。目前graph.node_groups里只存在复合操作节点组。
    for node_group_id in graph.node_groups:
        # 获取当前ID对应的节点组对象。
        node_group = graph.node_groups[node_group_id]
        # 检查当前节点组包含的节点数量是否大于1。只有当节点组内有多个节点时，才尝试通过DFS扩展它。
        if node_group.num_nodes() > 1:
            # 重置 adj_nodes 集合，为当前节点组的扩展做准备。
            adj_nodes = set()
            # 遍历当前节点组 (node_group) 中的每一个节点 (node)。
            for node in node_group.nodes.values():
                # 定义目标节点集 (dst_ids) 是 当前节点组中除了当前节点 (node) 之外的所有其他节点。这样做是为了在组内节点之间寻找连接路径。
                dst_ids = set(node_group.nodes.keys()).difference(set([node.id]))
                # 从当前节点 (node) 开始，执行DFS，试图找到连接到 dst_ids 中其他节点的路径。
                # graph.visited_dict() 应该返回一个所有节点都标记为未访问的字典。
                dfs_helper(graph, node, dst_ids, graph.visited_dict(), list())
            # DFS完成后，adj_nodes 中包含了所有在组内节点间路径上的节点。
            # 将这些找到的关联节点添加到原始的 node_group 中，实现节点组的扩展。
            node_group.add_nodes(adj_nodes)
            # 将扩展后的节点组添加到 node_groups 列表中。
            node_groups.append(node_group)
        # 记录下原始的节点组ID，无论它是否被扩展。
        old_node_group_ids.append(node_group_id)

    # 将旧的节点组从 graph.node_groups 中移除。这一步清空了 graph.node_groups 中原有的内容（如果这些组的ID被记录在 old_node_group_ids 中）。
    for old_node_group_id in old_node_group_ids:
        # 使用 del 操作符从字典中删除键值对。
        del graph.node_groups[old_node_group_id]
    
    # 遍历处理过的 node_groups 列表（包含了扩展后的节点组）。
    # 注意：如果一个原始的 node_group 节点数不大于1，它将不会出现在这个 node_groups 列表中，
    # 也因此不会被重新添加到 graph.node_groups 中，除非后续逻辑有其他处理。
    # 但由于 old_node_group_ids 包含了所有原始组的ID，这些未被处理的组也会被删除。
    # 这暗示了此函数可能期望 graph.node_groups 在调用前只包含那些需要被视为复合操作基础的、节点数大于1的组。
    for node_group in node_groups:
        # 调用节点组的 set_node_equivalence 方法，可能用于设置组内节点的等价关系或属性。
        node_group.set_node_equivalence()
        # 调用节点组的 set_output_nodes 方法，传入 graph 对象，用于确定并设置该节点组的输出节点。
        node_group.set_output_nodes(graph)
    # 返回处理后的节点组列表。这些节点组预期是 NodeGroupComposedOp 类型的基础。
    # 在 `build_pruning_dependency_graph` 中，这个返回值被赋给 `node_groups_composed_op`。
    return node_groups

def set_auxiliary_node_groups(graph):
    """
    处理辅助节点组（is_auxiliary），并将其与依赖的节点组建立关联。
    如果某辅助节点组依赖某些含 stem_op 的节点组，则将依赖关系注册到后者。
    参数:
        graph (Graph): 表示神经网络结构的图对象，包含节点、边及节点组等信息。
    """
    # 初始化一个字典，用于记录节点组是否已访问
    visited = dict()
    # 遍历图中的所有节点组
    for node_group in graph.node_groups.values():
        # 调用节点组的 set_auxiliary 方法，标记是否为辅助节点组
        if node_group.set_auxiliary():
            # 如果该节点组是辅助节点组，则标记为False
            visited[node_group.id] = False  # 初始化为未访问状态

    def dfs_helper(graph, node_group, dependent_node_groups):
        """
        深度优先搜索（DFS）辅助函数，用于递归查找当前辅助节点组所依赖的节点组。
        参数:
            graph (Graph): 当前处理的图对象。
            node_group (NodeGroup): 当前正在处理的节点组。
            dependent_node_groups (list): 用于存储依赖的节点组。
        """
        # 如果当前节点组不是辅助节点组，但包含 stem_op，则将其视为依赖节点组
        if not node_group.is_auxiliary:
            if node_group.contain_stem_op():
                dependent_node_groups.append(node_group)
            return
        # 如果当前节点组已访问过，则直接扩展其依赖的节点组
        elif visited[node_group.id]:
            if hasattr(node_group, 'dependent_node_groups'):
                dependent_node_groups.extend(node_group.dependent_node_groups)
            return

        # 获取当前节点组中的 concat 节点
        concat_nodes = node_group.get_concat_nodes()
        if len(concat_nodes) == 0:
            return

        # 取第一个 concat 节点，遍历其所有前驱节点
        concat_node = concat_nodes[0]
        for node_in in graph.incoming(concat_node):
            if node_in.id in ['dummy_input']:
                continue  # 跳过虚拟输入节点
            # 获取前驱节点所属的节点组
            node_group_in = graph.node_groups[node_in.node_group_ids[0]]
            if node_group_in.id != node_group.id:
                # 递归调用，继续查找依赖的节点组
                dfs_helper(graph, node_group_in, dependent_node_groups)

    # 遍历所有节点组，处理辅助节点组的依赖关系
    for node_group in graph.node_groups.values():
        if node_group.is_auxiliary:
            if visited[node_group.id]:
                continue  # 如果已访问过，跳过
            node_group.dependent_node_groups = list()  # 初始化依赖的节点组列表
            dfs_helper(graph, node_group, node_group.dependent_node_groups)

    # 对于 stem 节点组和辅助节点组的关联性，若辅助节点组无法找到依赖则视作非辅助
    for node_group in graph.node_groups.values():
        if node_group.is_auxiliary:
            if len(node_group.dependent_node_groups) == 0:
                # 如果辅助节点组没有依赖的节点组，则将其标记为非辅助
                node_group.is_auxiliary = False
                continue
            # 如果辅助节点组没有携带参数，则无需进一步处理
            if len(node_group.param_names) == 0:
                continue
            offset = 0  # 偏移量，用于记录辅助节点组在依赖节点组中的位置
            for depend_node_group in node_group.dependent_node_groups:
                if not hasattr(depend_node_group, 'auxilary_node_groups'):
                    depend_node_group.auxilary_node_groups = list()  # 初始化辅助节点组列表
                # 将辅助节点组及其偏移量添加到依赖节点组的辅助节点组列表中
                depend_node_group.auxilary_node_groups.append((node_group, offset))
                offset += depend_node_group.num_groups  # 更新偏移量

def merge_depth_conv_node_groups(graph):
    """
    合并深度可分离卷积（depthwise convolution）的节点组。
    如果一个卷积节点满足 groups = in_channels = out_channels，则表明是深度可分离卷积。
    在这种情况下，检查其父节点，若满足条件则合并相关的节点组。
    参数:
        graph (Graph): 表示神经网络结构的图对象，包含节点、边及节点组等信息。
    """
    # 初始化一个字典，用于记录每个节点是否已访问
    visited = dict()
    for node in graph.nodes.values():
        visited[node.id] = False  # 所有节点初始状态为未访问

    def dfs_helper(node, groups, node_groups_to_merge):
        """
        深度优先搜索（DFS）辅助函数，用于从当前节点出发，递归查找满足条件的父节点，并将其节点组加入合并列表。
        参数:
            node (Node): 当前处理的节点。
            groups (int): 当前卷积节点的 groups 值，用于判断是否满足深度可分离卷积条件。
            node_groups_to_merge (list): 用于存储需要合并的节点组。

        返回:
            list: 更新后的 node_groups_to_merge 列表。
        """
        if node.is_conv():
            # 如果当前节点是卷积节点，并且满足 groups = 1 且 out_channels = groups
            if node.op.module.groups == 1 and node.op.module.out_channels == groups:
                # 将该节点所属的节点组加入合并列表
                node_groups_to_merge.append(graph.node_groups[node.node_group_ids[0]])
                return node_groups_to_merge
        # 遍历当前节点的所有父节点（前驱节点）
        for node_in in graph.incoming(node):
            if not visited[node_in.id]:  # 如果父节点尚未访问
                visited[node_in.id] = True  # 标记为已访问
                dfs_helper(node_in, groups, node_groups_to_merge)  # 递归调用
        return node_groups_to_merge

    # 遍历图中的所有节点
    for node in graph.nodes.values():
        if node.is_conv():  # 如果当前节点是卷积节点
            # 检查该卷积节点是否具有 groups 属性
            if hasattr(node.op.module, 'groups'):
                # 如果满足 groups = in_channels，则认定为深度可分离卷积
                if node.op.module.groups == node.op.module.in_channels:
                    # 重置所有节点的访问状态为未访问
                    for node_id in graph.nodes:
                        visited[node_id] = False
                    # 初始化合并列表，包含当前节点所属的节点组
                    node_groups_to_merge = dfs_helper(
                        node, node.op.module.groups, [graph.node_groups[node.node_group_ids[0]]]
                    )
                    # 如果找到多个需要合并的节点组
                    if len(node_groups_to_merge) > 1:
                        # 获取第一个节点组的 ID（作为合并后的节点组 ID）
                        dummy_node_group_id = node_groups_to_merge[0].id

                        # --- 开始添加的代码 ---
                        # group_to_merge_into = node_groups_to_merge[0]
                        # group_being_merged = node_groups_to_merge[1]
                        # print(f"[合并信息] 在 merge_depth_conv_node_groups 函数中：\
                        #       尝试将节点组 ID {group_being_merged.id} (包含节点: {[n.id for n in group_being_merged.nodes.values()]}) \
                        #         合并到节点组 ID {group_to_merge_into.id} (包含节点: {[n.id for n in group_to_merge_into.nodes.values()]})")
                        # print("\n")

                        # 将第二个节点组合并到第一个节点组中
                        node_groups_to_merge[0].merge(node_groups_to_merge[1])
                        # 从图中删除旧的节点组
                        del graph.node_groups[dummy_node_group_id]
                        del graph.node_groups[node_groups_to_merge[1].id]
                        # 更新图中的节点组字典，使用合并后的节点组
                        graph.node_groups[node_groups_to_merge[0].id] = node_groups_to_merge[0]
                        # 更新合并后节点组中所有节点的 group ID
                        for node in node_groups_to_merge[0]:
                            node.node_group_ids[0] = node_groups_to_merge[0].id

def merge_basic_composed_node_groups(graph):
    """
    如果基础节点组（NodeGroup）与某复合节点组（NodeGroupComposedOp）拥有相同的 param_names，
    则合并这两者，最终作为一个组合算子节点组。
    参数:
        graph (Graph): 表示神经网络结构的图对象，包含节点、边及节点组等信息。
    """
    # 初始化一个字典，用于存储所有复合节点组（NodeGroupComposedOp），键为节点组 ID，值为节点组对象
    composed_node_groups = dict()
    for node_group in graph.node_groups.values():
        if type(node_group).__name__ == 'NodeGroupComposedOp':
            composed_node_groups[node_group.id] = node_group

    # 初始化两个列表：
    # - new_composed_node_groups：存储合并后的复合节点组
    # - merged_node_group_ids：存储已被合并的节点组 ID
    new_composed_node_groups = list()
    merged_node_group_ids = list()

    # 遍历所有基础节点组（NodeGroup）
    for node_group in graph.node_groups.values():
        if type(node_group).__name__ == 'NodeGroup':
            # 遍历所有复合节点组
            for composed_node_group_id in composed_node_groups:
                # 如果当前复合节点组已被合并，则跳过
                if composed_node_group_id in merged_node_group_ids:
                    continue
                # 否则获取当前复合节点组对象
                composed_node_group = composed_node_groups[composed_node_group_id]
                # 如果基础节点组和复合节点组的 param_names 集合相同，则可以合并
                if set(node_group.param_names) == set(composed_node_group.param_names):
                    # 将基础节点组和复合节点组的 ID 添加到已合并列表中
                    merged_node_group_ids.append(node_group.id)
                    merged_node_group_ids.append(composed_node_group_id)
                    # 遍历基础节点组中的每个节点
                    for node in node_group:
                        # 如果节点已经在复合节点组中，则跳过
                        if node.id in composed_node_group.nodes:
                            continue
                        # 否则，将节点添加到复合节点组中
                        composed_node_group.add_node(node)
                    # 将更新后的复合节点组添加到 new_composed_node_groups 列表中
                    new_composed_node_groups.append(composed_node_group)

    # 从图中移除已合并的旧节点组
    for node_group_id in merged_node_group_ids:
        del graph.node_groups[node_group_id]

    # 将合并后的复合节点组重新添加到图中
    for node_group in new_composed_node_groups:
        graph.node_groups[node_group.id] = node_group

    # 重新设置每个节点的 node_group_ids 属性
    for node in graph.nodes.values():
        node.node_group_ids = list()  # 清空节点的 node_group_ids 列表

    # 遍历所有节点组，为每个节点重新分配其所属的节点组 ID
    for node_group in graph.node_groups.values():
        # 遍历节点组中的所有节点
        for node in node_group:
            # 为节点添加节点组 ID
            node.node_group_ids.append(node_group.id)

def build_pruning_dependency_graph(graph):
    """
    主函数，构建可剪枝依赖关系图并对节点组执行各类合并、扩展、标记操作，以确定哪些节点组可剪枝。
    该函数是确定模型中哪些部分可以被剪枝的核心逻辑。
    它通过一系列步骤来分析计算图（graph对象），将节点组织成不同的组（NodeGroup），
    然后根据这些组的特性（如是否包含可训练参数、是否连接到输出、是否包含特定类型的操作等）
    以及它们之间的依赖关系，来最终决定每个组是否是“可剪枝的”（is_prunable）。
    参数:
        graph (Graph): 一个 Graph 对象，表示已经构建好的神经网络计算图。
                       该对象包含了节点、边、参数等信息。
    主要步骤:
    0.  预处理 复合算子节点组：识别由多个基础操作组成的复合操作，并将它们分组。
        这些组内的节点（除了输出节点）在后续步骤中可能会被跳过，以避免重复处理或错误分组。
    1.  获取非 stem 节点：识别图中那些不是“主干”（stem）的节点。Stem 节点通常指模型的早期层或关键层，可能不适合剪枝。同时排除之前在步骤0中标记为跳过的节点。
    2.  获取非 stem 节点组：将上一步得到的非 stem 节点根据它们在图中的连通性划分成不同的组。每个组代表一个连通分量。
    3.  扩展非 stem 节点组：对每个非 stem 节点组进行扩展，尝试将与组内节点紧密相关的 stem 节点（或其他特定类型的节点，如 concat 操作）也包含进来。
        扩展过程会避免包含在 skip_node_ids 中的节点。
    4.  合并节点组：如果不同的节点组之间存在共享的节点（即有交集），则将这些组进行合并，形成更大的节点组。
    5.  处理剩余节点：将在上述步骤中未能被分到任何组的独立节点，各自打包成只包含单个节点的组。
    6.  更新图的节点组信息：将所有形成的节点组（合并后的组、复合算子组、单节点组）添加到 `graph.node_groups` 字典中进行统一管理。
        同时，更新每个节点对象的 `node_group_ids` 属性，记录它所属的节点组ID。
    7.  合并深度可分离卷积节点组：特定地处理深度可分离卷积（depthwise separable convolutions）的模式。
        如果检测到这种模式，相关的节点组可能会被合并，因为它们的剪枝行为通常是耦合的。
    8.  设置辅助节点组：识别并标记“辅助”（auxiliary）节点组。辅助节点组通常是一些支持性的操作（如某些归一化层或激活函数），
        它们本身可能不直接参与主要的计算流，但依赖于其他核心节点组。此步骤还会建立辅助节点组与其所依赖的核心节点组之间的关联。
    9.  合并基础节点组与复合节点组：如果一个基础节点组（通常只包含简单操作）和某个复合节点组操作的是完全相同的参数集，则将它们合并。这有助于更准确地表示剪枝单元。
    10. 标记节点组的可剪枝性：这是最终决定哪些组可以被剪枝的关键步骤。
        -   首先，根据一些基本规则进行初步标记：
            -   如果一个节点组没有任何可训练参数，则标记为不可剪枝。
            -   如果一个节点组包含了模型的任何输出节点，则标记为不可剪枝。
            -   如果一个节点组既不包含任何 stem 操作，也不是一个辅助节点组，则标记为不可剪枝。
        -   然后，处理参数共享的情况：如果某些参数同时被一个可剪枝组和一个不可剪枝组共享，那么为了安全起见，将那个可剪枝组也标记为不可剪枝。
        -   接着，检查节点组中是否包含明确定义为不可剪枝的特定操作类型（例如，某些特殊的激活函数或自定义层，定义在 UNPRUNABLE_COMPOSED_OPERATORS 和 
            UNPRUNABLE_BASIC_OPERATORS 列表中）。如果包含，则该组不可剪枝。
        -   特殊情况处理：如果模型的虚拟输入节点（dummy_input）的输出直接通过加法（add）或乘法（mul）操作连接到一个节点组，那么这个节点组通常与输入缩放或偏置有关，标记为不可剪枝。
        -   覆写变换和分组数：处理一些特殊情况，如 Group Normalization。如果一个节点组内存在需要特定“传播”（spread）变换的参数（例如，GroupNorm 的 gamma 和 beta 参数需要根据分组数进行调整），
            则会统一该组内相关节点的参数变换方式（p_transform）和分组数（num_groups）。如果一个组内检测到多种不兼容的传播变换，会抛出错误。
        -   最后，针对辅助节点组的特殊规则：如果一个辅助节点组包含 Group Normalization 操作，并且其分组数大于1，那么它所依赖的那些核心节点组将被标记为不可剪枝。这是因为
            GroupNorm 的参数是按组共享的，剪枝其依赖的组可能会破坏这种结构。
    """
    '''# Step 0: 对复合算子节点组进行预处理'''
    # 调用 group_nodes_composed_operator 函数，识别并组合图中的复合操作节点。
    # 复合操作通常指那些由多个基础操作构成的、但在概念上可以视为一个单元的操作（例如，一个卷积层后面跟着批归一化和激活函数）。
    # 返回值 node_groups_composed_op 是一个列表，包含了这些复合算子节点组。
    node_groups_composed_op = group_nodes_composed_operator(graph)

    # 初始化一个空集合 skip_node_ids，用于存储在后续步骤中应该被忽略的节点ID。
    skip_node_ids = set()
    internal_nodes_to_skip = set()

    '''# Step 1: 获取非 stem 节点列表，排除 skip_node_ids'''
    # 调用 get_non_stem_nodes 函数，获取图中所有不属于 "stem" 部分的节点。
    # "stem" 通常指模型初始的几层，它们可能具有特殊的结构或对模型性能至关重要，因此在剪枝时可能需要特殊处理或不进行剪枝。
    # skip_node_ids 参数确保了在步骤0中标记的节点不会被包含在这个非stem节点列表中。
    non_stem_nodes = get_non_stem_nodes(graph, skip_node_ids=skip_node_ids)

    '''# Step 2: 获取非 stem 节点组成的连通分量组'''
    # 调用 get_non_stem_node_groups 函数，将上一步得到的 non_stem_nodes 列表中的节点，
    # 根据它们在计算图中的连通性，划分成若干个节点组。
    # 每个节点组代表图中的一个连通分量，即组内的节点可以通过图中的边相互到达。
    # 这里的节点组只包含非主干节点
    non_stem_node_groups = get_non_stem_node_groups(graph, non_stem_nodes)

    '''# Step 3: 扩展节点组，将必需的 stem 节点等也拉入当前组'''
    # 调用 grow_non_stem_node_groups 函数，对在步骤2中形成的每个非stem节点组进行扩展。
    # 扩展的目的是将与这些组紧密相关的 "stem" 节点或其他关键节点（如执行拼接操作的节点）也包含进来。
    # 这样做可以确保剪枝单元的完整性。skip_node_ids 同样用于避免包含不应被扩展的节点。
    grown_node_groups = grow_non_stem_node_groups(graph, non_stem_node_groups, skip_node_ids)

    '''# Step 4: 合并有交集的节点组'''
    # 调用 merge_node_groups 函数，处理在步骤3中扩展后可能产生的节点组之间的重叠。
    # 如果两个或多个节点组共享了至少一个相同的节点，那么这些组将被合并成一个更大的节点组。
    # 这个过程会持续进行，直到所有存在交集的组都被合并完毕。
    merged_node_groups = merge_node_groups(grown_node_groups)

    '''# Step 5: 将剩余的独立节点打包成单节点组'''
    # 调用 get_remaining_nodes 函数，找出在经过前面所有分组和合并步骤后，仍然没有被分配到任何节点组的节点。
    # 这些通常是图中的孤立节点或未被之前逻辑覆盖的小片段。
    # merged_node_groups + node_groups_composed_op 包含了目前已经形成的所有主要节点组。
    # graph.nodes 包含了图中所有的节点。
    remaining_nodes = get_remaining_nodes(merged_node_groups + node_groups_composed_op, graph.nodes)

    # 调用 group_individual_nodes 函数，将这些剩余的独立节点各自包装成一个只包含单个节点的 NodeGroup 对象。
    singleton_node_groups = group_individual_nodes(remaining_nodes)

    '''# Step 6: 将所有节点组加入 graph.node_groups 管理，并更新各节点的 node_group_ids'''
    # 遍历所有已经形成的节点组，包括：
    # - merged_node_groups: 经过合并步骤形成的主要节点组。
    # - node_groups_composed_op: 在步骤0中识别的复合算子节点组。
    # - singleton_node_groups: 在步骤5中为剩余独立节点创建的单节点组。
    for node_group in merged_node_groups + node_groups_composed_op + singleton_node_groups:
        # 将每个节点组以其ID为键，节点组对象本身为值，存入 graph 对象的 node_groups 字典中。
        # graph.node_groups 是图对象中用于集中管理所有节点组的容器。
        graph.node_groups[node_group.id] = node_group
        # 遍历当前节点组 (node_group) 中的每一个节点 (node)。
        for node in node_group:
            # 将当前节点组的ID (node_group.id) 添加到该节点对象的 node_group_ids 列表中。
            # 一个节点可能属于多个逻辑上的组（尽管在这里通常主要关联到一个），所以 node_group_ids 是一个列表。
            node.node_group_ids.append(node_group.id)

    '''# Step 7: 合并深度可分离卷积节点组'''
    # 调用 merge_depth_conv_node_groups 函数，专门处理图中可能存在的深度可分离卷积结构。
    # 深度可分离卷积由一个深度卷积（depthwise convolution）和一个逐点卷积（pointwise convolution）组成。
    # 这类结构的剪枝通常需要将这两个部分视为一个整体，因此该函数会尝试识别并合并代表这两部分的节点组。
    merge_depth_conv_node_groups(graph)

    '''# Step 8: 设置辅助节点组并建立依赖关联'''
    # 调用 set_auxiliary_node_groups 函数，识别图中的“辅助”节点组。
    # 辅助节点组通常包含一些本身不直接进行主要计算，但为其依赖的核心计算组提供支持的操作
    # （例如，某些类型的归一化层、激活函数，或者用于连接不同分支的拼接操作）。
    # 此函数还会建立这些辅助节点组与其所依赖的核心节点组之间的依赖关系。
    set_auxiliary_node_groups(graph)

    '''# Step 9: 若基础节点组能并入复合节点组，则进行合并'''
    # 调用 merge_basic_composed_node_groups 函数。
    # 这个函数检查是否存在一个“基础”节点组（通常只包含一个简单的、非复合的操作）和一个“复合”节点组
    # 它们操作的是完全相同的参数集。如果存在这种情况，意味着这个基础操作实际上是该复合操作的一部分，
    # 因此将这个基础节点组合并到相应的复合节点组中，以形成更完整的剪枝单元。
    merge_basic_composed_node_groups(graph)
    
    '''# Step 10: 标记节点组是否可剪枝'''
    # 遍历 graph.node_groups 字典中的所有节点组。
    for node_group in graph.node_groups.values():
        # 若节点组没有任何可训练的参数（即 node_group.param_names 为空），
        # 则该组不包含任何可以被剪枝的权重，因此将其标记为不可剪枝。
        if len(node_group.param_names) == 0:
            node_group.is_prunable = False
        # 若该节点组包含了图的任何一个输出节点（graph.output_nodes.values()），
        # 则剪枝该组可能会影响模型的最终输出，因此将其标记为不可剪枝。
        if node_group.contain_some_nodes(graph.output_nodes.values()):
            node_group.is_prunable = False
        # 如果一个节点组既不包含任何 "stem" 操作（通常是模型前几层或关键操作），
        # 也不是一个辅助节点组（node_group.is_auxiliary 为 False），
        # 那么它可能是一个独立的、非核心的计算片段，或者其剪枝规则不明确，因此也标记为不可剪枝。
        # Stem 操作的缺失可能意味着它不是一个主要的、可独立剪枝的计算单元。
        if not node_group.contain_stem_op() and not node_group.is_auxiliary:
            node_group.is_prunable = False

    # 如果某些权重同时属于可剪枝组和不可剪枝组，则设置成不可剪枝
    # 初始化一个空集合 unprunable_param_names，用于存储所有不可剪枝节点组中的参数名称。
    unprunable_param_names = set()
    # 遍历所有节点组。
    for node_group in graph.node_groups.values():
        # 如果当前节点组被标记为不可剪枝。
        if not node_group.is_prunable:
            # 将该不可剪枝组的所有参数名称添加到 unprunable_param_names 集合中。
            unprunable_param_names = unprunable_param_names.union(set(node_group.param_names))

    # 再次遍历所有节点组。
    for node_group in graph.node_groups.values():
        # 如果当前节点组最初被认为是可剪枝的。
        if node_group.is_prunable:
            # 检查当前可剪枝组的参数集合 (set(node_group.param_names)) 与所有不可剪枝参数的集合 (unprunable_param_names) 是否有交集。
            # 如果有交集，意味着这个“可剪枝”组中的某些参数也存在于某个“不可剪枝”的上下文中。
            # 为了保证模型的正确性，这种情况下需要将该组也标记为不可剪枝。
            if set(node_group.param_names) & unprunable_param_names:
                node_group.is_prunable = False

    # 检测是否包含不可剪枝的复合/基础算子
    # 遍历所有节点组。
    for node_group in graph.node_groups.values():
        # 检查当前节点组的类型名称是否为 'NodeGroupComposedOp'，即是否为复合算子节点组。
        if type(node_group).__name__ == 'NodeGroupComposedOp':
            # 如果是复合算子组，检查其主要操作 (node_group.op) 的类型名称是否存在于UNPRUNABLE_COMPOSED_OPERATORS 列表中。这个列表定义了哪些复合操作类型是不可剪枝的。
            if type(node_group.op).__name__ in UNPRUNABLE_COMPOSED_OPERATORS:
                # 如果包含不可剪枝的复合操作，则将该节点组标记为不可剪枝。
                node_group.is_prunable = False
        else: 
            # 如果不是复合算子组，即为基础节点组。
            # 遍历该基础节点组中的每一个节点。
            for node in node_group:
                # 检查当前节点的操作名称 (node.op_name) 是否存在于UNPRUNABLE_BASIC_OPERATORS 列表中。这个列表定义了哪些基础操作类型是不可剪枝的。
                if node.op_name in UNPRUNABLE_BASIC_OPERATORS:
                    # 如果包含不可剪枝的基础操作，则将该节点组标记为不可剪枝，并跳出内部循环。
                    node_group.is_prunable = False

    # 如果 dummy_input 后直接加/乘到节点组，则标记其不可剪枝
    # 获取图中的虚拟输入节点 'dummy_input'。
    dummy_input_node = graph.nodes['dummy_input']
    # 遍历从 dummy_input_node 出发的所有直接下游节点 (node_out)。
    for node_out in graph.outgoing(dummy_input_node):
        # 检查下游节点的操作名称是否为 'add' (加法) 或 'mul' (乘法)。
        if node_out.op_name == 'add' or node_out.op_name == 'mul':
            # 如果是加法或乘法，获取该下游节点所属的第一个节点组的ID。
            # 假设 node_out.node_group_ids[0] 存在并且是其主要的关联组。
            node_group_id = node_out.node_group_ids[0]
            # 将该节点组标记为不可剪枝。
            # 这种情况通常意味着输入数据直接参与了某种缩放或偏置操作，剪枝相关参数可能不安全。
            graph.node_groups[node_group_id].is_prunable = False

    # 覆写节点组的分组数和 p_transform（比如 group norm 等需要特别处理的情况）
    # 遍历所有节点组。
    for node_group in graph.node_groups.values():
        # 初始化一个集合 overwrite_p_transforms，用于存储需要被覆写的参数变换类型。
        overwrite_p_transforms = set()
        # 初始化 overwrite_num_groups 为0，用于存储需要被覆写的分组数。
        overwrite_num_groups = 0
        # 初始化一个集合 fixed_node_ids，用于存储那些其参数变换和分组数已经确定的节点的ID。
        fixed_node_ids = set()
        # 遍历当前节点组中的每一个节点。
        for node in node_group:
            # 如果节点没有参数，或者没有关联的操作对象 (node.op)，则跳过。
            if len(node.param_names) == 0 or not node.op:
                continue
            # 获取当前节点操作的参数分组信息，包括参数变换类型 (p_transform) 和分组数 (num_groups)。
            node_param_groups = node.op.get_param_groups(param_names=node.param_names)
            # 遍历该节点参数的所有变换类型。
            for p_transform in node_param_groups['p_transform']:
                # 检查当前的参数变换是否为一种“传播”类型的变换 (is_spread_transformation)。
                # 传播类型的变换通常意味着参数的结构（如分组）会影响到其他相关参数，例如 Group Normalization。
                if is_spread_transformation(p_transform):
                    # 如果是传播变换，将其添加到 overwrite_p_transforms 集合中。
                    overwrite_p_transforms.add(p_transform)
                    # 记录下由该传播变换确定的分组数。
                    overwrite_num_groups = node_param_groups['num_groups']
                    # 将当前节点的ID添加到 fixed_node_ids，表示这个节点的变换和分组数是固定的基准。
                    fixed_node_ids.add(node.id)

        # 如果在整个节点组中只找到一种确定的传播变换类型。
        if len(overwrite_p_transforms) == 1:
            # 获取这唯一的传播变换类型。
            # iter(overwrite_p_transforms) 会返回一个迭代器，指向集合中的第一个元素。
            # next(...) 则从迭代器中取出这个唯一的元素。
            overwrite_p_transform = next(iter(overwrite_p_transforms))
            # 再次遍历节点组中的所有节点。
            for node in node_group:
                # 如果节点没有参数，没有操作对象，或者是之前已标记为固定的节点，则跳过。
                if len(node.param_names) == 0 or not node.op or node.id in fixed_node_ids:
                    continue
                # 将这些节点的 num_groups 强制设置为之前确定的 overwrite_num_groups。
                node.op.num_groups = overwrite_num_groups
                # 将这些节点的 p_transform 强制设置为与基准传播变换相对应的标准变换形式。
                # SPREAD_TRANSFORM_MAP 可能是一个映射，将特定的传播变换名映射到一个标准的变换枚举或名称。
                node.op.p_transform = SPREAD_TRANSFORM_MAP[overwrite_p_transform]
            # 在节点组级别记录下这个被覆写的传播变换类型。
            node_group.overwrite_p_transform = overwrite_p_transform
        # 如果在节点组中找到了多种不同的传播变换类型，这意味着存在冲突，目前不支持这种情况。
        elif len(overwrite_p_transforms) > 1:
            raise NotImplementedError('One node group has two distinct spread_p_transforms.')

    # 当节点组是辅助节点组且包含 group norm（groups>1）时，将其依赖的节点组设为不可剪枝
    # 遍历所有节点组。
    for node_group in graph.node_groups.values():
        # 如果当前节点组不是辅助节点组，则跳过。
        if not node_group.is_auxiliary:
            continue

        # 遍历当前辅助节点组中的每一个节点。
        for node in node_group:
            # 如果节点没有参数或没有操作对象，则跳过。
            if len(node.param_names) == 0 or not node.op:
                continue
            # 检查节点的操作类型是否为 'GroupNormOTO'（自定义的 Group Normalization 操作）。
            if type(node.op).__name__ == 'GroupNormOTO':
                # 如果是 GroupNormOTO，并且其分组数 (num_groups) 大于1。
                if node.op.num_groups > 1:
                    # 遍历该辅助节点组所依赖的所有核心节点组 (node_group.dependent_node_groups)。
                    # 这些依赖关系是在 set_auxiliary_node_groups 函数中建立的。
                    for depend_node_group in node_group.dependent_node_groups:
                        # 将这些被依赖的核心节点组标记为不可剪枝。
                        # 这是因为 GroupNorm (当 groups > 1 时) 的参数（gamma 和 beta）是按通道组共享的，
                        # 如果剪枝了其依赖的特征图的通道，可能会导致 GroupNorm 的分组结构不匹配或失效。
                        depend_node_group.is_prunable = False

