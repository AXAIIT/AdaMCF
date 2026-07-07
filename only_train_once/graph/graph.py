import torch
from torch import _C
import numpy as np
import inspect
from packaging.version import Version
from collections import defaultdict
from only_train_once.operation import COMPOSED_MODULES, BASIC_MODULES, Operator, ParamOTO
from only_train_once.assets import THEMES
from .node import Node
from .node_group import NodeGroupComposedOp, NodeGroup, BasicNodeGroup
from only_train_once.transform.graph_transform import FRAMEWORK_TRANSFORMS
from only_train_once.transform import TensorTransform, tensor_transformation, index_transformation
import warnings
import pprint


# 检查torch版本，设置ONNX导出opset版本，保证兼容性
if Version(torch.__version__) >= Version('1.13.0'):
    from torch.onnx._globals import GLOBALS
    # 1.13默认opset 14不支持gridsample，这里强制设置为16
    GLOBALS.export_onnx_opset_version = 16 

from .utils import (
    _get_str_inside_parenthesis, 
    _optimize_trace_graph_no_onnx_operator, 
    _get_tensor_shape,
    _scale_value
)

class Graph:
    """跟踪有向图的节点和边，并支持基本操作。"""
    def __init__(self, model=None, dummy_input=None, trace_onnx=True, skip_patterns=None, strict_out_nodes=False):
        """
        初始化 Graph 对象，构建神经网络的 有向计算图结构 。
        参数说明：
        - model: 需要分析的 PyTorch 模型
        - dummy_input: 用于跟踪模型的输入样例（必须提供）
        - trace_onnx: 是否使用 ONNX 优化跟踪
        - skip_patterns: 跳过的模式（字符串或字符串列表），用于移除特定结构
        - strict_out_nodes: 是否严格只将无出边节点作为输出节点
        初始化内容：
        - 初始化节点、边、节点组、输入输出节点等数据结构
        - 处理参数分组（可训练/不可训练）
        - 解析 skip_patterns 参数
        - 构建模型的计算图结构
        - 应用图变换
        """
        print("OTO graph constructor")
        self.inputs = dict()            # 输入节点张量信息
        self.nodes = dict()             # 所有节点
        self.edges = list()             # 所有边
        self.node_groups = dict()       # 节点组
        self.output_nodes = dict()      # 输出节点
        self.input_nodes = dict()       # 输入节点
        self.dummy_input = dummy_input  # 追踪图用的输入
        self.params_grad = dict()       # 可训练参数
        self.params_no_grad = dict()    # 不可训练参数
        self.param_names = list()       # 参数名列表
        self.trace_onnx = trace_onnx    # 是否用 ONNX trace
        self.root_module = None         # 根模块
        self.theme = THEMES["basic"]    # 可视化主题
        self.skip_patterns = []         # 跳过的模式，跳过哪些节点和边

        # 处理 跳过模式 参数，支持字符串或字符串列表
        if skip_patterns is not None:
            # isinstance 用于检查 skip_patterns 是否是字符串类型
            if isinstance(skip_patterns, str):
                # 如果传入的是字符串，转为单元素列表，方便统一处理
                self.skip_patterns = [skip_patterns]
            # 检查 skip_patterns 是否是列表
            elif isinstance(skip_patterns, list):
                try:
                    # 检查列表中的每个元素是否都是字符串
                    assert(all([isinstance(a, str) for a in skip_patterns]))
                except:
                    # 如果不是字符串，抛出异常
                    raise ValueError("skip_patterns only supports string or list of strings")
                # 如果检查通过，直接赋值
                self.skip_patterns = skip_patterns
            else:
                # 如果既不是字符串也不是字符串列表，抛出异常
                raise ValueError("skip_patterns only supports string or list of strings")
        
        # 是否严格只把无出边节点作为输出节点
        self.strict_out_nodes = strict_out_nodes
        if not model:
            # 如果没有传入模型，直接返回，不进行后续操作
            return 
        
        self._model = model
        self.set_param_grad_no_grad(self._model)  # 设置参数分组（可训练/不可训练参数）
        
        # 检查 dummy_input 是否提供，若未提供则报错
        assert dummy_input is not None, "Dummy_input args must be provided for Pytorch models."
        model = model.eval()            # 设置模型为 eval 模式，保证推理时行为一致
        self.build(model, dummy_input)  # 构建计算图结构
        
        # 如果有 skip_patterns，移除指定结构（如某些算子组合）
        if len(self.skip_patterns) > 0:
            # 应用 skip_patterns，移除指定结构
            self.remove_patterns(self.skip_patterns)
        
        # 应用框架级别的图变换（如节点重命名、节点融合、结构优化等）
        for t in FRAMEWORK_TRANSFORMS:
            # 对每一个框架级别的图变换对象t，调用其apply方法，这里的self就是当前的Graph实例
            # 具体的变换内容由FRAMEWORK_TRANSFORMS列表中的对象决定，这里只使用 节点重命名 
            t.apply(self)

        print("--- self.output_nodes   所有输出节点：---")
        pprint.pprint(self.output_nodes)
        print("\n")

        print("--- self.input_nodes    所有输入节点：---")
        pprint.pprint(self.input_nodes)
        print("\n")

        print("--- self.dummy_input (type and shape/keys if applicable)   dummy input的来源: ---")
        if isinstance(self.dummy_input, torch.Tensor):
            print(f"Type: torch.Tensor, Shape: {self.dummy_input.shape}")
        elif isinstance(self.dummy_input, (list, tuple)):
            print(f"Type: {type(self.dummy_input)}")
            for i, t_in in enumerate(self.dummy_input):
                if isinstance(t_in, torch.Tensor):
                    print(f"  Item {i}: Type: torch.Tensor, Shape: {t_in.shape}")
                else:
                    print(f"  Item {i}: Type: {type(t_in)}")
        elif isinstance(self.dummy_input, dict):
            print(f"Type: dict, Keys: {list(self.dummy_input.keys())}")
            for k, v_in in self.dummy_input.items():
                if isinstance(v_in, torch.Tensor):
                    print(f"  Key '{k}': Type: torch.Tensor, Shape: {v_in.shape}")
                else:
                    print(f"  Key '{k}': Type: {type(v_in)}")
        else:
            print(f"Type: {type(self.dummy_input)}, Value: {self.dummy_input}")
        print("\n")

        print(f"--- self.trace_onnx，是否使用trace_onnx优化：{self.trace_onnx}  ---")
        print("\n")

        print("--- self.skip_patterns ---   所有跳过的模式：---")
        pprint.pprint(self.skip_patterns)
        print("\n")

    def build(self, model, dummy_input):
        """
        构建计算图的主方法。
        1. 获取trace graph。
        2. 解析模型模块，建立操作与参数的映射。
        3. 遍历trace graph节点，生成Node对象，建立节点与边。
        4. 标记输入、输出节点，补充dummy节点便于可视化。
        5. 可选地将部分matmul节点替换为linear节点，清理孤立节点。
        6. 处理节点组（如卷积、线性层等），建立节点组与边的关系。
        7. 处理参数分组（可训练/不可训练），建立参数与节点的关系。
        8. 处理节点组的参数分组（可训练/不可训练），建立参数与节点组的关系。
        9. 处理节点组的输入输出关系，建立节点组与输入输出的关系。
        10. 处理节点组的参数关系，建立节点组与参数的关系。
        参数：
        - model: 需要分析的 PyTorch或onnx 模型
        - dummy_input: 用于跟踪模型的输入样例（必须提供）
        返回：
        - None
        """
        print("graph build")
        # 1. 获取trace graph（追踪计算图）
        trace_graph = self._get_trace_graph(model, dummy_input, optimized_onnx=self.trace_onnx)

        '''
        # 2. 解析模型模块，建立操作与参数的映射
        - state_dict (dict): 包含模型参数的字典，用于确定参数张量的数量。
        - torch_graph_str (str): PyTorch计算图的字符串表示，用于解析张量信息。
        '''
        self._parse_modules(model)
    
        '''
        # 3. 解析张量信息的方法，提取输入、参数等信息。
        - model.state_dict(): 获取模型的状态字典，包含所有参数和缓冲区。
        - str(trace_graph): 将trace graph转换为字符串表示，便于解析。
        '''
        self._parse_tensors_info(model.state_dict(), str(trace_graph))
        
        '''
        # defaultdict创建了一个默认值为 set 的字典。
        # 它的特点是当访问的键不存在时，会自动为该键创建一个默认值，而不会抛出 KeyError 异常。
        graph = defaultdict(set)
        graph['A'].add('B')
        graph['A'].add('C')
        graph['B'].add('D')
        print(graph)
        # 输出: defaultdict(<class 'set'>, {'A': {'B', 'C'}, 'B': {'D'}})
        '''
        torch_nodes_by_inputs = defaultdict(set)   # 记录每个输入张量对应的节点
        torch_nodes_by_outputs = defaultdict(set)  # 记录每个输出张量对应的节点
        
        # 4. 遍历trace graph(追踪计算图)中的每个节点，构建Node对象
        # 事实上就是遍历graph(…………):之后的每一个计算节点，详细见输出的trace_graph
        print("遍历trace graph中的每个节点，构建Node对象:   ")
        for torch_node in trace_graph.nodes():
            '''
            # 获取操作名
            # 通过节点的 kind() 方法获取该节点的操作类型字符串
            # kind() 方法返回的字符串格式为 "aten::conv2d" 或 "prim::Constant" 等
            # 这里我们只关心操作名部分，去掉前面的 "aten::" 或 "prim::" 前缀
            # 通过 split("::") 方法将字符串分割成列表，[-1]取最后一个元素
            # 然后将下划线 "_" 替换为空字符串 ""，并转换为小写字母
            # 最终得到操作名
            '''
            op_name = torch_node.kind().split("::")[-1].lower().replace('_', '')

            '''
            # 获取操作配置参数
            # 通过节点的 attributeNames() 方法获取该节点的所有属性名，例如卷积层的 kernel_size、stride、padding 等
            # 然后使用 getattr() 方法获取每个属性的值
            # 最后将每个属性值转换为字符串，并保存到一个字典中
            # 最终得到操作参数
            '''
            op_cfg_params = {k: getattr(torch_node, torch_node.kindOf(k))(k) for k in torch_node.attributeNames()}

            '''
            # 获取输出shape
            # _get_tensor_shape(...):
            # 这是一个自定义的辅助函数（定义在 graph/utils.py 中）。
            # 它的作用是从包含形状信息的字符串中解析出实际的形状元组。
            # 例如，从 "Float(2, 3, 224, 224)" 这样的字符串中提取出 (2, 3, 224, 224)。
            # .split(':'):将节点的字符串表示按照冒号 ":" 分割。
            # .strip():去掉分割后字符串的前后空格。
            # [1]:取分割后列表的第二个元素（索引为 1）。
            '''
            output_shape = _get_tensor_shape(str(torch_node).split(':')[1].strip())
    
            # 提取其所有输入张量和输出张量的唯一标识符
            inputs = [i.unique() for i in torch_node.inputs()]
            outputs = [o.unique() for o in torch_node.outputs()]
    
            '''
            # 获取参数名
            # 获取当前节点（torch_node）所有输入张量的真实参数名称列表
            # 1. `inputs`: 这是一个列表，包含了当前 `torch_node` 所有输入张量在计算图中的唯一ID（整数）。
            # 2. `for i in inputs`: 遍历 `inputs` 列表中的每一个张量ID `i`。
            # 3. `if i in self.param_id_to_name`: 这是一个条件判断。
            #    `self.param_id_to_name` 是一个字典，它是在 `_parse_tensors_info` 方法中构建的，
            #    用于将计算图中参数张量的唯一ID映射到它们在模型中的实际名称（例如 "conv1.weight", "fc.bias"）。
            #    这个条件检查当前的输入张量ID `i` 是否存在于 `self.param_id_to_name` 字典的键中。
            #    如果存在，说明这个输入张量 `i` 实际上是模型的一个参数（而不是模型的外部输入或中间激活值）。
            # 4. `self.param_id_to_name[i]`: 如果条件为真（即 `i` 是一个参数张量的ID），
            #    则通过这个表达式从 `self.param_id_to_name` 字典中获取该参数ID `i` 对应的实际参数名称（字符串）。
            # 5. `[...]`: 这是一个列表推导式。它将所有满足条件（即属于模型参数）的输入张量的实际名称收集起来，
            #    形成一个新的列表。
            # 最终，`param_names` 列表将包含当前 `torch_node` 所使用的所有模型参数的名称。
            # 如果 `torch_node` 不使用任何模型参数（例如，它是一个激活函数节点，只处理来自其他节点的激活值），
            # 则 `param_names` 将是一个空列表。
            '''
            param_names = [self.param_id_to_name[i] for i in inputs if i in self.param_id_to_name]
    
            # 构建Operator（操作）对象
            op = None
            if len(param_names) > 0:
                # 如果当前节点有参数名
                if param_names[0] in self.param_name_to_operator:
                    op = self.param_name_to_operator[param_names[0]]    # 获取操作对象（包含复合操作和基本操作）
                    op.cfg_params = op_cfg_params                       # 更新Operator的配置参数
                elif len(param_names) == 1 and param_names[0] in self.params_grad:
                    '''
                    # 如果只有一个参数名，且该参数是可训练参数，则构建ParamOTO对象（带参数的Operator）
                    # _type：操作类型名（如conv2d、linear等）
                    # cfg_params：操作的配置参数（如kernel_size等）
                    # param_name：参数名
                    # param：参数张量
                    '''
                    op = ParamOTO(_type=op_name, cfg_params=op_cfg_params, param_name=param_names[0], param=self.params_grad[param_names[0]])
                else:
                    # 其它情况（如多个参数或不可训练参数），构建普通Operator对象
                    # _type：操作类型名（如conv2d、linear等）
                    # cfg_params：操作的配置参数（如kernel_size等）
                    op = Operator(_type=op_name, cfg_params=op_cfg_params)
                    for param_name in param_names:
                        # 遍历所有参数名，将参数张量加入 Operator 的 name_to_param 字典
                        # 判断该参数名是否在可训练参数字典中
						# 如果是可训练参数，则取 self.params_grad[param_name]
						# 否则取不可训练参数 self.params_no_grad[param_name]
                        op.name_to_param[param_name] = self.params_grad[param_name] if param_name in self.params_grad else self.params_no_grad[param_name]
            else:
                # 如果没有参数名，直接构建普通Operator对象
                op = Operator(_type=op_name, cfg_params=op_cfg_params)
    
            # 创建Node对象
            node = Node(
                id=self.torch_node_id(torch_node),   # 节点唯一ID，通常格式为"node-xxx"，由PyTorch trace节点的输出唯一标识生成 + node- 构成
                op_name=op_name,                     # 操作名（如conv2d、linear、relu等），用于标识节点类型
                op=op,                               # Operator（操作）对象，封装了该节点的算子类型、参数、配置等信息
                inputs=inputs,                       # 输入张量的唯一ID列表（即本节点依赖的上游节点的输出）
                outputs=outputs,                     # 输出张量的唯一ID列表（即本节点产生的输出）
                param_names=param_names,             # 该节点涉及的参数名列表（如权重、偏置等），便于参数追踪和剪枝
                output_shape=output_shape            # 节点输出张量的shape信息，用于可视化和FLOPs计算等
            )

            # 如果属于复合操作节点组，加入节点组
            if op.id in self.op_name_to_node_group_comp_op:
                # 如果当前Operator属于复合操作（对应预定义复合操作节点），
                # 则将该节点加入对应的节点组（NodeGroupComposedOp），便于后续分组剪枝等操作
                self.op_name_to_node_group_comp_op[op.id].add_node(node)
            
            # 添加节点到图: self.nodes
            # 将当前节点对象加入到Graph的nodes字典中，便于后续查找和遍历
            self.add_node(node)

            # 记录边关系（输出张量与节点的对应关系）
            # 遍历当前节点的所有输出张量
            for output in outputs:
                torch_nodes_by_outputs[output].add(torch_node)      # 记录该输出张量由当前节点产生
                for target_torch_node in torch_nodes_by_inputs[output]:
                    # 如果有其他节点以该输出为输入，则建立边（当前节点 -> 目标节点）
                    self.add_edge_by_id(self.torch_node_id(torch_node), self.torch_node_id(target_torch_node))

            # 记录边关系（输入张量与节点的对应关系）
            for input in inputs:
                # 遍历当前节点的所有输入张量
                torch_nodes_by_inputs[input].add(torch_node)        # 记录该输入张量被当前torch_node使用
                for target_torch_node in torch_nodes_by_outputs[input]:
                    # 如果有其他节点产生该输入，则建立边（目标节点 -> 当前节点）
                    self.add_edge_by_id(self.torch_node_id(target_torch_node), self.torch_node_id(torch_node))
    
        '''# 5. 将复合操作节点组加入node_groups节点组''' 
        for op_id in self.op_name_to_node_group_comp_op:
            # 遍历所有复合操作（对应预定义复合操作节点）的节点组，将其加入Graph的node_groups字典
            # 这样后续可以通过node_groups统一管理所有节点组（包括复合操作组和普通组），便于剪枝、可视化、参数分组等操作
            node_group = self.op_name_to_node_group_comp_op[op_id]
            self.node_groups[node_group.id] = node_group

        # 6. 处理输出节点（无出边的节点为输出节点）
        # 初始化一个集合 out_ids，用于存储被识别为输出节点的节点ID。使用集合可以自动处理重复的ID。
        out_ids = set()
        # 遍历 self.nodes 字典中的所有节点对象。self.nodes 是一个字典，键是节点ID，值是 Node 对象。
        for node in self.nodes.values():
            # 调用 self.outgoing(node) 方法获取从当前节点 node 出发的所有下游节点。len(self.outgoing(node)) == 0 判断当前节点是否没有任何出边。
            if len(self.outgoing(node)) == 0:
                # 如果该节点没有任何出边（即没有下游节点），则认为它是图的一个输出节点。将该节点的 ID (node.id) 添加到 out_ids 集合中。
                out_ids.add(node.id)
        # 检查 self.strict_out_nodes 标志。这个标志在 Graph 初始化时设置，用于控制如何确定输出节点。
        if not self.strict_out_nodes:
            # 如果不要求严格只把“无出边节点”作为输出节点（即 self.strict_out_nodes 为 False），那么还可以直接使用 PyTorch JIT 追踪图 (trace_graph) 本身声明的输出作为补充。
            # 遍历 trace_graph.outputs() 返回的每个输出值对象。
            # trace_graph.outputs() 返回的是计算图在 JIT 层面定义的输出张量列表。
            for out in trace_graph.outputs():
                # 将输出值对象 out 转换为字符串，并按空格分割，取第一个元素。
                # PyTorch JIT 图中，输出张量的字符串表示通常以 "%<id>" 开头，例如 "%123 : Float(1, 10)"。
                # str(out).split()[0] 会提取出 "%<id>" 这部分。
                out_id = str(out).split()[0]
                # trace_graph 的输出张量ID格式为 "%xxx" (例如 "%123")。
                # 为了与 Graph 内部节点ID ("node-xxx") 的格式保持一致，
                # 这里将 "%" 去掉（通过 out_id[1:]，如果确定是%开头的话，或者直接拼接），然后在前面加上 "node-" 前缀。
                out_ids.add('node-' + out_id)       # 将格式化后的输出节点ID添加到 out_ids 集合中。
    
        # 7. 标记整张图的输入输出节点
        # 遍历 self.nodes 字典中的所有节点对象（节点值）。
        for node in self.nodes.values():
            # 检查当前节点的 ID 是否在 out_ids 集合中。
            if node.id in out_ids:
                # 如果节点id在输出节点集合out_ids中，则将该节点标记为输出节点，加入self.output_nodes字典
                self.output_nodes[node.id] = node
            if len(set(node.inputs).intersection(set(self.inputs.keys()))) > 0:
                '''
                # 检查当前节点的 输入张量ID集合 与图的整体输入张量ID集合是否有交集
                # node.inputs 是一个列表，包含了当前节点所有输入张量的ID（这些ID是Graph内部统一格式化的，例如 'node-1', 'node-2'）
                # set(node.inputs) 将其转换为一个集合，便于进行集合运算（如交集）
                # self.inputs 是一个字典，在 _parse_tensors_info 方法中填充，
                # 其键是图的实际输入张量的ID（例如 'node-0', 'node-1'，对应于dummy_input中的张量）
                # set(self.inputs.keys()) 获取这个字典所有键的集合
                # .intersection(...) 计算两个集合的交集，即找出同时存在于 node.inputs 和 self.inputs.keys() 中的张量ID
                # len(...) > 0 判断交集是否非空。
                # 如果交集非空，意味着当前节点的至少一个输入张量是整个计算图的原始输入之一。
                '''
                # 如果节点的输入张量与模型输入张量有交集，则将该节点标记为输入节点，加入self.input_nodes字典
                self.input_nodes[node.id] = node
                
    
        # 8. 设置每个节点的输入shape
        # 遍历 self.nodes 字典中的所有节点对象
        for node in self.nodes.values():
            if len(node.inputs) == 0:
                # 如果没有输入张量，则跳过该节点，例如：常量节点、输入节点等
                continue  
            # 获取所有指向该节点的上游节点
            nodes_in = self.incoming(node)  
            if len(nodes_in) == 0:
                # 如果没有上游节点，说明输入来自模型的原始输入张量
                for in_id in node.inputs:
                    # 遍历当前节点的所有输入张量ID
                    if in_id not in self.inputs:
                        # 如果输入张量不在整个模型输入张量中，则跳过该张量
                        continue  
                    # 获取模型输入张量的shape
                    input_shape = _get_tensor_shape(self.inputs[in_id][-1][1], prefix_str='Float')
                    # 将模型输入张量的shape加入当前节点的input_shape
                    node.input_shape.append(input_shape)  
            else:
                # 如果有上游节点，则将所有上游节点的output_shape作为当前节点的input_shape
                for node_in in nodes_in:
                    # 将上游节点的output_shape加入当前节点的input_shape
                    node.input_shape.append(node_in.output_shape)

                    
        # 9. 添加dummy输入输出节点，便于可视化
        # 创建两个虚拟节点，dummy_input_node 和 dummy_output_node，dummy_input_node 用于表示模型的输入，dummy_output_node 用于表示模型的输出
        dummy_input_node = Node(id='dummy_input', op_name='dummy_input')        
        dummy_output_node = Node(id='dummy_output', op_name='dummy_output')
        self.add_node(dummy_input_node)     # 将虚拟输入节点添加到图中
        self.add_node(dummy_output_node)    # 将虚拟输出节点添加到图中
        for input_node in self.input_nodes.values():
            # 为每个输入节点添加一条来自dummy_input的边，便于可视化时突出模型输入
            self.add_edge(dummy_input_node, input_node)
        for output_node in self.output_nodes.values():
            # 为每个输出节点添加一条指向dummy_output的边，便于可视化时突出模型输出
            self.add_edge(output_node, dummy_output_node)
            
        
        # 10. 进一步处理matmul等节点，优化为linear，移除孤立节点
        if self.trace_onnx:
            '''
            # 如果使用ONNX优化，可能会有一些matmul节点被优化为linear节点，这里的处理是为了统一图结构，便于后续剪枝和可视化：
            # 1. 将满足条件的matmul节点转换为linear节点（如有转置权重和加偏置），便于结构统一和剪枝
            # 2. 移除无输入且无参数的孤立节点，保证图结构简洁
            '''
            self.replace_eligible_matmul_as_linear()    # 将符合条件的matmul节点转换为linear节点
            self.remove_isolated_nodes()                # 移除无输入且无参数的孤立节点，保证图结构简洁
            
    def remove_patterns(self, skip_patterns):
        # 警告：该方法不会保证图结构的有效性，使用时需谨慎
        # 按深度优先遍历顺序移除路径模式，skip_patterns: 需要移除的路径模式列表，每个模式为字符串，如 "a->b->c"
        # 例如：模型结构为 "input->conv->bn->conv->bn->conv->output"，移除 "conv->bn->conv" 后只剩下 input 和 output 节点且两者不再连通
        warnings.warn("This method does not gurantee the validity of the graph. Users should be careful when using this method.")
        all_remove_nodes = [] # 初始化一个列表，用于存储每个模式查找到的所有匹配路径。每个路径本身是一个节点ID列表。
        for pattern in skip_patterns: 
            # 遍历用户传入的每个待移除的 模式字符串
            # 对每个待跳过的模式，查找所有匹配的节点路径，收集待移除节点
            nodes_path_to_remove = self._find_remove_pattern(pattern)   # 调用内部辅助方法 _find_remove_pattern 查找当前模式匹配的所有路径
            all_remove_nodes.append(nodes_path_to_remove)               # 将找到的路径列表（可能为空）添加到 all_remove_nodes 中
        # 展开所有路径，收集所有需要移除的节点id
        all_remove_nodes_unique = []                                    # 初始化一个列表，用于收集所有路径中的 所有节点ID（此时可能包含重复ID）
        for pattern in all_remove_nodes:
            # 遍历all_remove_nodes，其中每个元素是对应一个模式找到的所有路径的列表
            for found_path in pattern: 
                # 遍历当前模式下的每一条匹配路径（found_path 是一个节点ID列表）
                for node in found_path:
                    # 遍历当前路径中的每一个节点ID
                    all_remove_nodes_unique.append(node) # 将节点ID添加到列表中
        # 去重，得到所有需要移除的唯一节点id
        all_remove_nodes_unique = list(set(all_remove_nodes_unique)) # 利用set的特性去除重复的节点ID，然后转换回列表
        for node_to_remove in all_remove_nodes_unique: 
            # 遍历所有唯一的待移除节点ID
            # 从节点字典中移除该节点
            self.nodes.pop(node_to_remove) # 从图的节点字典 self.nodes 中移除该节点
            # 如果是输入节点，也从输入节点字典中移除
            if node_to_remove in self.input_nodes: 
                # 检查该待移除节点是否也记录在图的输入节点字典 self.input_nodes 中
                self.input_nodes.pop(node_to_remove) # 如果是，则也从输入节点字典中移除
            # 如果是输出节点，也从输出节点字典中移除
            if node_to_remove in self.output_nodes: # 检查该待移除节点是否也记录在图的输出节点字典 self.output_nodes 中
                self.output_nodes.pop(node_to_remove) # 如果是，则也从输出节点字典中移除
        # 重新构建边，只保留未被移除节点之间的边
        edges_new = [] # 初始化一个新的边列表，用于存储更新后的边
        for edge in self.edges: # 遍历图中当前所有的边
            # 如果边的起点或终点在移除节点中，则跳过
            # edge[0] 是边的起点ID，edge[1] 是边的终点ID
            if edge[0] in all_remove_nodes_unique or edge[1] in all_remove_nodes_unique: 
                # 检查边的起点或终点是否在待移除节点列表中
                continue # 如果是，则这条边也应该被移除（因为它连接了已被移除的节点），所以跳过
            else:
                edges_new.append(edge) # 否则，这条边连接的两个节点都未被移除，将其保留到新的边列表中
        self.edges = edges_new # 更新图的边列表为新的边列表
        # 查找并移除所有与主图断开的孤立节点
        disconnected_nodes = self._find_disconnected_nodes() # 调用内部辅助方法 _find_disconnected_nodes 查找所有与主图输出不连通的节点
        for disconnected_node in disconnected_nodes: 
            # 遍历所有找到的孤立节点ID
            # 从节点字典中移除孤立节点
            self.nodes.pop(disconnected_node) # 从图的节点字典 self.nodes 中移除该孤立节点
            # 如果是输入节点，也从输入节点字典中移除
            if disconnected_node in self.input_nodes:       # 检查该孤立节点是否也记录在图的输入节点字典中
                self.input_nodes.pop(disconnected_node)     # 如果是，则也从输入节点字典中移除
            # 如果是输出节点，也从输出节点字典中移除
            if disconnected_node in self.output_nodes:      # 检查该孤立节点是否也记录在图的输出节点字典中
                self.output_nodes.pop(disconnected_node)    # 如果是，则也从输出节点字典中移除
        # 再次清理边，移除与孤立节点相关的边
        edges_new = [] # 再次初始化一个新的边列表
        for edge in self.edges: # 遍历图中（可能已因移除模式节点而更新过的）所有边
            # 如果边的起点或终点在孤立节点中，则跳过
            if edge[0] in disconnected_nodes or edge[1] in disconnected_nodes: # 检查边的起点或终点是否在已移除的孤立节点列表中
                continue # 如果是，则这条边也应该被移除，所以跳过
            else:
                edges_new.append(edge)  # 否则，保留这条边
        self.edges = edges_new          # 最终更新图的边列表，移除了与孤立节点相关的边

    def _find_remove_pattern(self, pattern):
        """
        查找所有与给定路径模式匹配的节点路径，用于后续批量移除。
        参数:
            pattern: 路径模式字符串，如 "conv->bn->conv"
        返回:
            nodes_path_to_remove: 所有匹配该模式的节点路径列表，每个路径为节点id列表
        """
        # 将模式字符串按"->"分割，得到节点类型序列（列表）
        pattern_node_names = pattern.split("->")
        
        def _dfs_helper(node, node_names):
            """
            递归DFS辅助函数，从当前节点出发，查找是否存在与 node_names 匹配的路径。
            参数:
                node: 当前遍历到的节点对象
                node_names: 剩余待匹配的节点类型序列
            返回:
                所有匹配路径的节点id列表（二维列表），如[[id1, id2, id3], ...]
            """
            # 初始化一个列表，用于存储当前节点匹配到的所有路径
            remove_nodes = []
            # 边界条件：节点为空或无操作名，直接返回None
            if node is None or node.op_name is None:
                return None
            
            # 如果当前节点类型与模式首节点类型匹配
            if node.op_name == node_names[0]:
                outgoing_nodes = self.outgoing(node)  # 获取所有下游节点
                nodes_child = node_names[1:]          # 剩余待匹配的节点类型序列
    
                if len(nodes_child) == 0:
                    # 如果已匹配到模式末尾，返回当前节点id作为完整路径
                    return [[node.id]]
    
                if len(outgoing_nodes) > 0:
                    # 如果还有下游节点，递归查找所有下游节点是否能继续匹配剩余模式
                    for child in outgoing_nodes:
                        # 递归调用，查找下游节点是否能匹配剩余模式
                        marked_nodes = _dfs_helper(child, nodes_child)
                        if marked_nodes is not None:
                            # 如果下游节点匹配成功，遍历所有匹配路径
                            for marked_node in marked_nodes:
                                # 拼接当前节点id与下游匹配路径
                                remove_nodes.append([node.id] + marked_node)
                    return remove_nodes
                else:
                    # 没有下游节点但模式未匹配完，返回None
                    return None
            else:
                # 当前节点类型与模式不匹配，返回None
                return None
        # 初始化一个列表，用于存储所有找到的匹配路径
        nodes_path_to_remove = []
        # 遍历图中的所有节点，查找与模式匹配的路径
        for node in self.nodes.values():
            # 递归调用，查找当前节点是否能匹配模式
            marked_nodes_path = _dfs_helper(node, pattern_node_names)
            if marked_nodes_path is not None:
                # 合并所有找到的匹配路径
                nodes_path_to_remove = nodes_path_to_remove + marked_nodes_path
        return nodes_path_to_remove
    
    def _find_disconnected_nodes(self):
        """
        查找所有与主图输出节点不连通的孤立节点。
        返回：
            disconnected_nodes: 孤立节点的id列表（这些节点无法通过有向路径到达任何输出节点）
        主要用途：
            - 在结构剪枝、模式移除等操作后，自动清理与主图断开的无用节点，保证图结构简洁。
        实现思路：
            - 对每个节点，递归DFS判断其是否能通过有向路径到达任意输出节点。
            - 如果不能到达任何输出节点，则视为孤立节点。
        """
        visited_connected = set()  # 记录已确认连通到输出节点的节点id，避免重复递归
    
        def _dfs_helper(node):
            """
            递归DFS辅助函数，判断当前节点是否能到达任意输出节点。
            参数:
                node: 当前遍历到的节点对象
            返回:
                True  - 当前节点能到达输出节点
                False - 当前节点无法到达输出节点
            """
            outgoing_nodes = self.outgoing(node)  # 获取所有下游节点
            if len(outgoing_nodes) == 0: 
                # 如果没有下游节点，说明是叶子节点，无法到达输出节点
                return False
            if node.id in self.output_nodes:
                # 如果当前节点本身就是输出节点，直接返回True
                return True
            if node.id in visited_connected:
                # 如果当前节点已被确认连通到输出节点，直接返回True，避免重复递归
                return True
            
            connected = False   # 初始化标记，表示当前节点是否能到达输出节点
            for child in outgoing_nodes:
                # 递归判断所有下游节点是否能到达输出节点
                connected = (connected or _dfs_helper(child))
                if connected:
                    # 只要有一个下游节点能到达输出节点，则当前节点也能到达
                    visited_connected.add(node.id)
            return connected
        disconnected_nodes = [] # 初始化一个列表，用于存储所有孤立节点的id
        for node in self.nodes.values():
            # 遍历图中的所有节点，判断每个节点是否能到达输出节点
            connected = _dfs_helper(node)   # 判断当前节点是否能到达输出节点
            if not connected:
                # 如果该节点无法到达任何输出节点，则加入孤立节点列表
                disconnected_nodes.append(node.id)
        return disconnected_nodes
                      
    def replace_eligible_matmul_as_linear(self):
        """
        将符合条件的 矩阵乘法（matmul）节点 替换为 线性（linear）节点 。
        该函数遍历图中的所有节点，找到符合条件的矩阵乘法节点，并将其替换为线性节点。
        符合条件的矩阵乘法节点通常与转置（transpose）节点和加法（add）节点相关联。
        替换后，转置节点和加法节点将被移除，其参数将被合并到线性节点中。
        该函数不返回任何值，直接修改图中的节点和边。
        替换逻辑：
        符合条件的 matmul 节点需要满足以下主要条件才能被替换为 linear 节点：
        1. 节点本身是 matmul 类型：node.op_name == 'matmul'。
        2. 存在一个特定的上游转置节点（transpose）作为其输入：
           这个上游输入节点的操作类型是 transpose (node_in.op_name == 'transpose')。
           这个 transpose 节点本身没有更上游的输入节点 (len(self.incoming(node_in)) == 0)。这通常意味着这个转置操作是直接作用于一个参数（如权重矩阵）。
        3. 此外，如果上述条件满足，代码还会 可选地 检查是否存在一个下游的加法节点（add）来合并偏置（bias）：
           存在一个特定的下游加法节点（add）作为其输出：
           这个下游输出节点的操作类型是 add (node_out.op_name == 'add')。
           这个 add 节点只有一个上游输入（即当前 matmul 节点的输出） (len(self.incoming(node_out)) == 1)。
        总结来说，一个 matmul 节点如果其权重输入是通过一个无输入的 transpose 节点提供的，那么它就符合被转换为 linear 节点的基本条件。
        如果它还有一个符合条件的下游 add 节点，那么这个 add 节点（通常代表偏置）也会被合并到新的 linear 节点中。
        """
        # 第一遍遍历：找到所有符合条件的矩阵乘法节点
        # 记录符合条件的matmul节点及其相关的转置和加法节点的列表
        matmul_nodes = list()
        # 遍历所有节点
        for node in self.nodes.values():
            # 如果不是矩阵乘法节点，则跳过
            if node.op_name != 'matmul':
                continue
            # 标记是否进行转换
            do_convert = False
            # 初始化 转置权重节点 和 加偏置节点 为None
            transpose_weight_node = None
            add_bias_node = None
    
            # 检查当前matmul节点的输入，是否有转置节点（且转置节点没有上游节点，说明是权重转置）
            # 遍历当前matmul节点的所有上游输入节点
            for node_in in self.incoming(node):
                # 如果上游节点是转置操作，并且该转置节点没有更上游的输入（表明它可能是直接作用于一个参数，如权重）
                if node_in.op_name == 'transpose' and len(self.incoming(node_in)) == 0:
                    do_convert = True               # 标记可以进行转换
                    transpose_weight_node = node_in # 记录这个转置节点
                    # 检查matmul节点的输出，是否有add节点（且add节点只有一个输入，说明是加bias）
                    # 遍历当前matmul节点的所有下游输出节点
                    for node_out in self.outgoing(node):
                        # 如果下游节点是加法操作，并且该加法节点只有一个上游输入（即matmul的输出）
                        # 这通常意味着这个加法操作是用来添加偏置（bias）的
                        if node_out.op_name == 'add' and len(self.incoming(node_out)) == 1:
                            add_bias_node = node_out # 记录这个加法节点
    
            # 如果满足转换条件（找到了权重转置节点）
            if do_convert:
                # 将符合条件的matmul节点及其关联的转置节点和（可能存在的）加法节点信息存入列表
                matmul_nodes.append(
                    {
                        'matmul': node,                             # 原始的matmul节点
                        'transpose_weight': transpose_weight_node,  # 关联的权重转置节点
                        'add_bias': add_bias_node                   # 关联的偏置加法节点 (可能为None)
                    }
                )
    
        # 第二遍遍历：将所有符合条件的matmul节点替换为linear节点
        removed_add_bias_nodes = set() # 用于记录已移除的加法节点，避免重复处理（虽然在此逻辑中可能不需要）
        # 遍历之前收集到的所有符合条件的matmul节点信息
        for node_dict in matmul_nodes:
            matmul_node = node_dict['matmul']                       # 获取原始matmul节点
            transpose_weight_node = node_dict['transpose_weight']   # 获取权重转置节点
            add_bias_node = node_dict['add_bias']                   # 获取偏置加法节点
    
            # 1. 将matmul节点类型改为linear，并继承转置节点的参数信息
            matmul_node.op_name = 'linear'                                  # 修改matmul节点的操作名称为'linear'
            matmul_node.op = transpose_weight_node.op                       # 将matmul节点的Operator对象替换为转置节点的Operator对象（通常包含权重参数）
            matmul_node.param_names = transpose_weight_node.param_names     # 将matmul节点的参数名列表替换为转置节点的参数名列表
    
            # 2. 从图中移除转置节点
            self.remove(transpose_weight_node) # 调用Graph的remove方法移除转置节点及其相关的边
            # 同步移除节点组中的转置节点，并将matmul节点（现在是linear节点）加入该节点组
            # 遍历所有复合操作的节点组
            for node_group in self.op_name_to_node_group_comp_op.values():
                # 如果节点组包含被移除的转置节点
                if node_group.contain_node(transpose_weight_node):
                    node_group.remove_node(transpose_weight_node)   # 从节点组中移除转置节点
                    node_group.add_node(matmul_node)                # 将转换后的linear节点（原matmul节点）加入该节点组
    
            # 3. 如果存在加法节点（bias），合并其参数到linear节点，并重定向边
            if add_bias_node is not None:
                # 将加法节点（偏置）的参数名追加到新的linear节点的参数名列表中
                matmul_node.param_names.extend(add_bias_node.param_names)
                # 将原add节点的所有下游节点，改为由新的linear节点（原matmul节点）指向
                # 遍历原加法节点的所有下游输出节点
                for node_out in self.outgoing(add_bias_node):
                    self.add_edge(matmul_node, node_out)    # 添加从新的linear节点到这些下游节点的边
                # 从图中移除add节点
                self.remove(add_bias_node)                  # 调用Graph的remove方法移除加法节点及其相关的边
                removed_add_bias_nodes.add(add_bias_node)   # 记录已移除的加法节点
                # 同步移除节点组中的add节点
                # 遍历所有复合操作的节点组
                for node_group in self.op_name_to_node_group_comp_op.values():
                    # 如果节点组包含被移除的加法节点
                    if node_group.contain_node(add_bias_node):
                        node_group.remove_node(add_bias_node) # 从节点组中移除加法节点
        
    def remove_isolated_nodes(self):
        """
        移除所有没有任何输入节点且没有参数的孤立节点（即无入边且无参数的节点）。
        主要用于清理图结构中无意义的孤立节点，保证图结构简洁。
        典型场景：如ONNX trace后可能会出现一些无用的辅助节点，这些节点既没有输入也没有参数，对剪枝和可视化无意义。
        实现思路：
        - 通过all_nodes_have_incoming辅助函数判断当前图中是否还存在 无入边且无参数 的节点。
        - 如果存在，则收集所有这类节点，批量移除。
        - 重复上述过程，直到所有节点都满足“有入边或有参数”。
        - dummy_input节点除外，不会被移除。
        """
        def all_nodes_have_incoming(graph):
            """
            判断图中是否所有节点都至少有一个入边或有参数（dummy_input节点除外）。
            返回True表示所有节点都满足条件，否则返回False。
            """
            # 初始化结果为True
            result = True
            # 遍历图中的所有节点
            for node in graph.nodes.values():
                if node.id == 'dummy_input':
                    # 跳过dummy输入节点
                    continue  
                # 如果节点没有入边且没有参数，则返回False
                if len(graph.incoming(node)) == 0 and len(node.param_names) == 0:
                    result = False
            return result
    
        # 迭代移除所有无入边且无参数的孤立节点
        while not all_nodes_have_incoming(self):
            # 创建列表，收集所有无入边且无参数的节点
            nodes_no_incoming = list()
            # 遍历图中的所有节点
            for node in self.nodes.values():
                if node.id == 'dummy_input':
                    # 如果节点是dummy输入节点，则跳过
                    continue 
                # 判断是否为无入边且无参数的节点
                if len(self.incoming(node)) == 0 and len(node.param_names) == 0:
                    nodes_no_incoming.append(node)
            # 批量移除所有孤立节点（包括相关边）
            self.remove(nodes_no_incoming)
    
    def add_node(self, node):
        """
        将一个节点对象添加到图的节点字典中。
        参数:
            node: Node对象，表示要添加到图中的节点。
        详细说明：
        - 该方法会为传入的节点分配唯一id（通过self.id(node)），
          并将其存入self.nodes字典，key为节点id，value为节点对象本身。
        - 这样可以保证每个节点在图中的唯一性，便于后续通过id快速查找、管理节点。
        - 如果节点id已存在，则会覆盖原有节点（通常不会发生，除非手动重复添加同一节点）。
        - 该方法是构建和维护有向计算图的基础操作之一，所有节点的增删查改都依赖于self.nodes字典。
        """
        node_id = self.id(node)     # 获取节点的唯一标识符
        self.nodes[node_id] = node  # 将节点对象存入节点字典

    def add_edge_by_id(self, vid1, vid2, label=None):
        """
        根据节点id在图中添加一条有向边。
        参数:
            vid1: 边的起点节点id（字符串）
            vid2: 边的终点节点id（字符串）
            label: 可选，边的标签（如用于可视化、区分不同类型的连接）
        详细说明：
        - 该方法直接将一条有向边（vid1 -> vid2）以三元组(vid1, vid2, label)的形式添加到self.edges列表中。
        - self.edges是一个存储所有边的列表，每条边由起点id、终点id和可选标签组成。
        - 该方法不会检查边是否已存在，可能会出现重复边（如需去重可在add_edge中处理）。
        - 主要用于在构建计算图时，通过节点id快速建立节点之间的连接关系，便于后续遍历、可视化和分析。
        """
        self.edges.append((vid1, vid2, label))

    def outgoing(self, node):
        """
        获取所有从给定节点（或节点列表）出发的出边所连接的下游节点对象列表。
        参数:
            node: 单个Node对象或Node对象列表，表示要查询的起始节点。
        返回:
            outgoing: 列表，包含所有与给定节点（组）有出边关系的下游节点对象。
        详细说明：
        - 支持传入单个节点或节点列表，统一处理。
        - 首先将输入节点统一为列表，并获取其所有节点id。
        - 遍历self.edges（所有边的三元组列表，[00, 11, 22]），筛选出所有起点在node_ids中且终点不在node_ids中的边，即“从该组节点出发，指向组外节点”的所有出边。
        - 对每条出边，取其终点id（e[1]），通过self[e[1]]获取对应的节点对象，组成返回列表。
        - 常用于查找某节点的所有下游节点，或某组节点的所有外部下游节点，便于图遍历、结构分析等。
        """
        # 确保 'node' 参数是一个列表，如果它只是单个节点对象，则将其包装在列表中。这样做是为了统一处理单个节点和节点列表的情况。
        nodes = node if isinstance(node, list) else [node]
        # 使用列表推导式，为 'nodes' 列表中的每个节点 'n' 调用 self.id(n) 方法获取其唯一ID，
        # 并将这些ID收集到一个新的列表 'node_ids' 中。self.id(n) 通常返回节点 'n' 的 'id' 属性。
        node_ids = [self.id(n) for n in nodes]
        # 找到所有从该组节点出发、指向组外节点的出边
        # 这是一个列表推导式，用于构建 'outgoing' 列表，其中包含所有下游节点对象。
        # 1. `for e in self.edges`: 遍历图中的每一条边 'e'。
        #    self.edges 是一个存储所有边的列表，每条边 'e' 通常是一个元组 (source_id, target_id, label)。
        # 2. `if e[0] in node_ids and e[1] not in node_ids`: 这是一个条件判断，用于筛选边。
        #    - `e[0] in node_ids`: 检查边的起始节点ID (e[0]) 是否在 'node_ids' 列表中。
        #      这确保了我们只考虑从输入节点（或节点组）出发的边。
        #    - `e[1] not in node_ids`: 检查边的目标节点ID (e[1]) 是否不在 'node_ids' 列表中。
        #      这确保了我们只考虑指向输入节点组外部的节点的边（即真正的“出边”到其他节点）。
        # 3. `self[e[1]]`: 如果上述条件满足，则 `e[1]` 是一个指向组外节点的下游节点的ID。
        #    `self[e[1]]` 使用 Graph 类的 `__getitem__` 方法（通过 `self.nodes.get(e[1])` 实现），
        #    根据目标节点ID `e[1]` 从 `self.nodes` 字典中获取对应的 节点对象。
        # 最终，'outgoing' 列表包含了所有从输入节点（组）出发，并连接到该组外部的下游节点对象。
        outgoing = [self[e[1]] for e in self.edges
                    if e[0] in node_ids and e[1] not in node_ids]
        return outgoing

    def incoming(self, node):
        """
        获取所有指向给定节点（或节点列表）的入边所连接的上游节点对象列表。
        参数:
            node: 单个Node对象或Node对象列表，表示要查询的目标节点。
        返回:
            incoming: 列表，包含所有与给定节点（组）有入边关系的上游节点对象。
        详细说明：
        - 支持传入单个节点或节点列表，统一处理。
        - 首先将输入节点统一为列表，并获取其所有节点id。
        - 遍历self.edges（所有边的三元组列表），筛选出所有终点在node_ids中且起点不在node_ids中的边，即“指向该组节点、但起点不在组内”的所有入边。
        - 对每条入边，取其起点id（e[0]），通过self[e[0]]获取对应的节点对象，组成返回列表。
        - 常用于查找某节点的所有上游节点，或某组节点的所有外部上游节点，便于图遍历、结构分析等。
        """
        # 确保 'node' 参数是一个列表，如果它只是单个节点对象，则将其包装在列表中。这样做是为了统一处理单个节点和节点列表的情况。
        nodes = node if isinstance(node, list) else [node]
        # 使用列表推导式，为 'nodes' 列表中的每个节点 'n' 调用 self.id(n) 方法获取其唯一ID，
        # 并将这些ID收集到一个新的列表 'node_ids' 中。self.id(n) 通常返回节点 'n' 的 'id' 属性。
        # 这里的 'node_ids' 列表包含了所有输入节点（或节点组）的唯一ID。
        node_ids = [self.id(n) for n in nodes]
        # 找到所有指向该组节点、但起点不在组内的入边
        # 这是一个列表推导式，用于构建 'incoming' 列表，其中包含所有上游节点对象。
        # 1. `for e in self.edges`: 遍历图中的每一条边 'e'。
        #    self.edges 是一个存储所有边的列表，每条边 'e' 通常是一个元组 (source_id, target_id, label)。
        # 2. `if e[1] in node_ids and e[0] not in node_ids`: 这是一个条件判断，用于筛选边。
        #    - `e[1] in node_ids`: 检查边的目标节点ID (e[1]) 是否在 'node_ids' 列表中。
        #      这确保了我们只考虑指向输入节点（或节点组）的边。
        #    - `e[0] not in node_ids`: 检查边的起始节点ID (e[0]) 是否不在 'node_ids' 列表中。
        #      这确保了我们只考虑起点不在输入节点组内的边（即真正的“入边”）。
        # 3. `self[e[0]]`: 如果上述条件满足，则 `e[0]` 是一个指向输入节点的上游节点的ID。
        #    `self[e[0]]` 使用 Graph 类的 `__getitem__` 方法（通过 `self.nodes.get(e[0])` 实现），
        #    根据起始节点ID `e[0]` 从 `self.nodes` 字典中获取对应的 节点对象。
        # 最终，'incoming' 列表包含了所有指向输入节点（组）并连接到该组外部的上游节点对象。
        incoming = [self[e[0]] for e in self.edges
                    if e[1] in node_ids and e[0] not in node_ids]
        return incoming

    def __getitem__(self, key):
        """
        根据给定的键获取节点值。
        参数:
        - key: 可以是单个键或键的列表。如果是一个列表，则返回对应键的节点值列表；如果是单个键，则返回对应的节点值。
        返回值:
        - 如果key是列表，则返回一个包含对应节点值的列表；如果key是单个键，则返回对应的节点值。如果键不存在，则返回None。
        """
        # 处理key为列表的情况，返回对应键的节点值列表
        if isinstance(key, list):
            return [self.nodes.get(k) for k in key]
        # 处理key为单个键的情况，返回对应的节点值
        else:
            return self.nodes.get(key)

    def add_edge(self, node1, node2, label=None):
        """
        在图中添加一条边。
        参数:
        - node1: 边的起始节点。
        - node2: 边的终止节点。
        - label: 边的标签，可选参数，默认为None。
        返回值:
        无返回值。如果边已经存在，则不会重复添加。
        # 如果边已经存在，则不重复添加
        # TODO: 如果边存在但标签不同，仍然不添加
        """
        # 获取边的起点和终点id
        edge = (self.id(node1), self.id(node2), label)
        # 检查边是否已经存在于图中
        # 如果边已经存在，则不重复添加
        # 这里的self.edges是一个列表，存储了所有边的三元组（起点id，终点id，标签）
        if edge not in self.edges:
            self.edges.append(edge)

    def remove(self, nodes):
        """
        从图中移除指定的节点及其相关的边。
        参数:
        nodes (单个节点或节点列表): 要移除的节点。可以是一个单独的节点，也可以是一个节点列表。
        返回值:
        无
        """
        # 如果传入的节点不是列表，则将其转换为列表，以便统一处理
        nodes = nodes if isinstance(nodes, list) else [nodes]
        # 遍历所有要移除的节点
        for node in nodes:
            # 获取节点的唯一标识符
            k = self.id(node)
            # 过滤掉所有与该节点相关的边
            self.edges = list(filter(lambda e: e[0] != k and e[1] != k, self.edges))
            # 从节点字典中删除该节点
            del self.nodes[k]
            
    def _get_trace_graph(self, model, dummy_input, optimized_onnx=False):
        """
        获取PyTorch模型的trace graph（追踪计算图）。
        支持多种dummy_input类型（dict、Tensor、tuple/list），并根据PyTorch版本选择不同的优化方式。
        参数:
            model: 需要trace的PyTorch模型
            dummy_input: 输入样例，可以是dict、Tensor或tuple/list
            optimized_onnx: 是否使用ONNX优化trace
        返回:
            trace_graph: 追踪得到的计算图对象
        """
        trace_graph = None
        # 使用torch.no_grad()上下文管理器，确保在追踪图的生成过程中不计算和存储梯度，这对于推理和图分析是标准做法。
        with torch.no_grad(): 
            # 处理不同类型的dummy_input，检查dummy_input是否为字典类型。
            if isinstance(dummy_input, dict): 
                # 如果输入为dict，按模型forward参数顺序组装输入
                # 使用 inspect.signature 获取模型 forward 方法的签名，进而得到其参数名列表。
                forward_args = inspect.signature(model.forward).parameters.keys()
                # 初始化一个列表，用于按 forward 方法参数顺序存放最终的输入张量。
                input_tensors = [] 
                print(f"模型forward方法的参数名:")
                # 遍历模型 forward 方法的每一个参数名。
                for argname in forward_args: 
                    # 排除通用的可变参数*args和**kwargs，因为它们不直接对应具名输入。
                    if argname not in ['args', 'kwargs']:  
                        # 如果当前参数名存在于用户提供的 dummy_input 字典中。
                        if argname in dummy_input:
                            input_tensor = dummy_input[argname]                                       # 从字典中获取该参数对应的输入张量。
                            input_tensors.append(input_tensor)                                        # 将获取到的张量添加到列表中。
                            print(f"参数名: {argname}, 对应的输入张量形状: {input_tensor.shape}")       # 打印参数名及其形状，通常用于调试。
                        else:
                            # 如果 forward 方法的某个参数在 dummy_input 字典中没有提供，则假定其为可选参数或有默认值，并在此处用 None 占位。
                            input_tensors.append(None)
                # 将输入张量列表转换为元组，因为 torch.jit._get_trace_graph 函数期望参数以元组形式提供。
                input_tensors = tuple(input_tensors)

            # 否则，如果 dummy_input 是一个单独的 PyTorch 张量。
            elif isinstance(dummy_input, torch.Tensor): 
                # 如果输入为单个Tensor，转为tuple，将这个单独的张量包装成一个单元素的元组。
                input_tensors = (dummy_input,) 
            # 其他情况，假定 dummy_input 已经是某种序列类型（例如列表或元组）。
            else: 
                # 其它情况（如list/tuple），直接转为tuple，直接将 dummy_input 转换为元组，以确保类型符合追踪函数的要求。
                input_tensors = tuple(dummy_input)
            # 获取trace graph，调用 PyTorch 内部的 _get_trace_graph 函数来执行模型的追踪，从而生成计算图。
            # model: 需要被追踪的 PyTorch 模型实例。
            # args=input_tensors: 经过处理后，符合模型 forward 方法签名的输入参数元组。
            # 该函数返回两个主要的值：第一个是预追踪图对象 (trace_graph)，并不是最终的追踪计算图，还要进一步优化，第二个是模型执行 dummy_input后的输出 (此处用 _ 符号忽略该输出)。
            trace_graph, _ = torch.jit._get_trace_graph(model, args=input_tensors)
            print("--- 最终传递给 torch.jit._get_trace_graph 的 input_tensors: ---")
            if isinstance(input_tensors, tuple):
                print(f"类型: tuple, 长度: {len(input_tensors)}")
                for i, tensor_item in enumerate(input_tensors):
                    if isinstance(tensor_item, torch.Tensor):
                        print(f"  元素 {i}: 类型: torch.Tensor, 形状: {tensor_item.shape}, 设备: {tensor_item.device}")
                    elif tensor_item is None:
                        print(f"  元素 {i}: None")
                    else:
                        print(f"  元素 {i}: 类型: {type(tensor_item)}")
            else:
                # 理论上此时 input_tensors 必定是元组，但作为防御性编程，处理其他可能性
                print(f"类型: {type(input_tensors)}, 内容: {input_tensors}")
            print("--------------------------------------------------------------------")

        # 不使用ONNX优化追踪的情况
        if not optimized_onnx:
            # 不优化时，调用自定义优化函数，去除ONNX算子
            trace_graph = _optimize_trace_graph_no_onnx_operator(trace_graph, torch.onnx.OperatorExportTypes.ONNX)
        else:
            # 根据PyTorch版本选择不同的ONNX优化方式
            # 截至2025.05.09，pytorch版本已经更新到2.6
            if Version(torch.__version__) >= Version('1.9.0') and Version(torch.__version__) <= Version('1.11.10'):
                print(f"Info: Used torch.onnx.utils._optimize_graph for PyTorch {torch.__version__}.")
                trace_graph = torch.onnx._optimize_trace(trace_graph, torch.onnx.OperatorExportTypes.ONNX)
            elif Version(torch.__version__) >= Version('1.13.0') and Version(torch.__version__) < Version('1.7.0'):
                print(f"Info: Used torch.onnx.utils._optimize_graph for PyTorch {torch.__version__}.")
                trace_graph = torch.onnx._optimize_graph(trace_graph, torch.onnx.OperatorExportTypes.ONNX)
            elif Version(torch.__version__) >= Version('1.7.0'): # 例如 PyTorch 2.7.0 及更新版本
                from torch.onnx import utils as onnx_utils
                print(f"Info: Used torch.onnx.utils._optimize_graph for PyTorch {torch.__version__}.")
                trace_graph = onnx_utils._optimize_graph(trace_graph, torch.onnx.OperatorExportTypes.ONNX)
            else:
                # 其它未覆盖的版本 (例如 1.12.x, 或低于 1.9.0 的版本)
                raise RuntimeError(
                    f"Torch version {torch.__version__} does not have a supported ONNX graph optimization path in this script. "
                    "The version falls into an unhandled range.")
        return trace_graph

    def _get_module_type(self, module):
        """
        获取模块的类型名称。
        参数:
            module: 一个PyTorch模块（nn.Module的实例）
        返回:
            str类型，模块的类名（如"Conv2d"、"Linear"等）
        用途说明：
        - 该方法用于递归遍历模型结构时，判断当前模块属于哪种类型。
        - 便于后续根据类型（如卷积、线性、归一化等）做不同的处理或分组。
        """
        return type(module).__name__

    def _parse_modules(self, model):
        """
        递归解析模型，建立基本操作和复合操作的映射关系，并建立参数名到操作的映射。
        主要作用：
        - 找到模型的根模块（包含全部参数的最外层模块）
        - 递归遍历模型，识别所有复合操作（如Sequential、Bottleneck等）和基本操作（如Conv、Linear等）
        - 建立 参数名 到 操作对象 的映射，便于后续节点构建和参数分组
        """
        # 初始化模型参数名集合
        model_param_names = set()
        # 收集模型所有参数名
        for name, _ in model.named_parameters():
            model_param_names.add(name)
        # print(model_param_names)
        
        for m in model.modules():
            # 遍历模型的所有子模块（包括模型本身和所有嵌套的子层）
            # 获取 PyTorch 模块 m 及其所有子模块中所有参数的名称，并将这些名称存储在一个集合（set）中，赋值给变量 module_param_names。
            module_param_names = set([name for name, _ in m.named_parameters()])
            # 收集当前模块（及其所有子模块）拥有的参数名，形成一个集合
            if module_param_names == model_param_names:
                # 如果当前模块拥有的参数名集合和整个模型的参数名集合完全一致，说明这个模块就是“根模块”（即包含了模型所有参数的最外层模块）
                self.root_module = m
                break  # 找到后立即退出循环
        print(f"模型的根模块为：\n{self.root_module}\n")
    
        self.basic_ops = dict()    # 存储所有基本操作对象
        self.composed_ops = dict() # 存储所有复合操作对象
        
        '''
        以下两个深度优先遍历寻找复合操作和基本操作都是在self.root_module下进行的，
        也就是说在self.root_module里的所有子模块都被遍历到。
        '''
        def find_compose_op_dfs_helper(module, module_name, composed_op):
            """
            深度优先递归查找 复合操作（如Sequential、Bottleneck等），并保存到self.composed_ops。
            参数说明：
            - module: 当前遍历到的PyTorch模块
            - module_name: 当前模块的名称（字符串，带有父模块前缀）
            - composed_op: 父级复合操作对象（递归时传递）
            主要逻辑：
            1. 判断当前模块类型是否属于复合操作（如Sequential、Bottleneck等）。
            - 如果是，则实例化对应的复合操作对象，并保存到self.composed_ops字典中，key为模块名。
            - 递归终止（return），不再向下遍历该模块的子模块。
            2. 如果不是复合操作，则递归遍历其所有子模块（named_children），并为每个子模块拼接好完整的模块名（父名.子名），继续递归查找。
            """
            # 获取当前模块的类型名称
            # 通过 _get_module_type 方法获取当前模块的类型名称（如Conv2d、Linear等）
            module_type = self._get_module_type(module)
            # 判断当前模块是否为复合操作类型，通过判断模块类型是否在预定义的复合操作类型列表中，来确定当前模块是否为复合操作
            if module_type in COMPOSED_MODULES:
                # 如果是复合操作类型（如Sequential、Bottleneck等），实例化并保存
                composed_op = COMPOSED_MODULES[module_type](
                    id = module_name,         # 复合操作的唯一名称（带前缀）
                    _type = module_type,      # 复合操作的类型名
                    module = module,          # 复合操作对应的PyTorch模块对象
                    model = self._model
                )
                self.composed_ops[composed_op.id] = composed_op  # 保存到复合操作字典，便于后续查找和分组
                return  # 递归终止：遇到复合操作后不再向下递归其子模块
            
            # 因为从根模块开始深度优先遍历的，所以所有模块都是根模块的子模块，递归遍历所有子模块
            for name, module_child in module.named_children():
                # 递归查找子模块是否为复合操作，module_name为空时不加点，否则加点拼接，保证每个子模块有唯一全名
                find_compose_op_dfs_helper(module_child, module_name + '.' + name if module_name != '' else name, composed_op)

        # 从根模块开始递归查找复合操作
        find_compose_op_dfs_helper(self.root_module, "", None)
    
        def find_basic_op_dfs_helper(module, module_name, basic_op):
            """
            深度优先递归查找 基本操作（如Conv、Linear等），并保存到self.basic_ops。
            """
            # 获取当前模块的类型名称
            # 通过 _get_module_type 方法获取当前模块的类型名称（如Conv2d、Linear等）
            module_type = self._get_module_type(module)
            if module_type in COMPOSED_MODULES:
                # 如果是复合操作，跳过
                return
            if module_type in BASIC_MODULES:
                # 如果是基本操作类型，实例化并保存
                basic_op = BASIC_MODULES[module_type](
                    id = module_name,
                    _type = module_type,
                    module = module)
                self.basic_ops[basic_op.id] = basic_op
                return 
            
            # 因为从根模块开始深度优先遍历的，所以所有模块都是根模块的子模块，递归遍历子模块
            for name, module_child in module.named_children():
                # 递归查找子模块是否为基本操作，module_name为空时不加点，否则加点拼接，保证每个子模块有唯一全名
                find_basic_op_dfs_helper(module_child, module_name + '.' + name if module_name != '' else name, basic_op)
            
        # 从根模块开始递归查找基本操作
        find_basic_op_dfs_helper(self.root_module, "", None)   
        
        # 建立参数名到操作对象的映射
        self.param_name_to_operator = dict()            # 参数名到操作对象（基本操作/复合操作）的映射
        self.op_name_to_node_group_comp_op = dict()     # 复合操作名到节点组对象的映射
        
        # 复合操作的参数名映射，遍历所有复合操作对象，获取其参数名，并建立映射关系，方便后续查找和操作。
        for op_name in self.composed_ops:
            # 获取当前复合操作对象
            compose_op = self.composed_ops[op_name]
            '''
            # 为每个复合操作创建一个节点组对象，便于后续分组和剪枝
            '''
            self.op_name_to_node_group_comp_op[op_name] = NodeGroupComposedOp(op=compose_op)
            # 遍历当前复合操作的参数，建立参数名到操作对象的映射，复合操作可以进步展开成更多基本操作，包含更多参数
            for p_name, _ in compose_op.named_parameters():
                # 复合操作的参数名格式为“复合操作名.参数名”，映射到对应的复合操作对象
                self.param_name_to_operator[op_name + '.' + p_name] = compose_op

        # 基本操作的参数名映射，遍历所有基本操作对象，获取其参数名，并建立映射关系，方便后续查找和操作。
        for op_name in self.basic_ops:
            # 获取当前基本操作对象
            basic_op = self.basic_ops[op_name]
            # 为每个基本操作创建一个节点组对象，便于后续分组和剪枝
            for p_name, _ in basic_op.named_parameters():
                # 基本操作的参数名格式为 “基本操作名.参数名” ，映射到对应的基本操作对象
                self.param_name_to_operator[op_name + '.' + p_name] = basic_op


    def id(self, node):
        """
        获取节点的唯一标识符。
        如果节点有 id 属性（推荐），则直接返回该 id；
        否则，返回节点对象的 hash 值作为唯一标识。
        这样可以保证每个节点在图中的唯一性，便于节点查找和边的管理。
        """
        return node.id if hasattr(node, "id") else hash(node)

    def torch_node_id(self, node):
        """
        为给定的节点生成一个唯一的标识符。
        该函数通过将节点的所有输出对象的唯一标识符连接起来，生成一个全局唯一的节点标识符。
        参数:
        - node: 需要生成唯一标识符的节点对象。该节点应具有 `outputs()` 方法，返回节点的输出对象列表。
        返回值:
        - str: 生成的唯一节点标识符，格式为 "node-<output1_unique_id>-<output2_unique_id>-..."。
        """
        # 将节点的所有 输出对象的唯一标识符 连接起来，生成 唯一的节点标识符
        return "node-" + "-".join(["{}".format(o.unique()) for o in node.outputs()])

    def _parse_tensors_info(self, state_dict, torch_graph_str):
        """
        解析PyTorch计算图中的张量信息，并将其分类为输入张量和参数张量。
        该函数通过解析PyTorch计算图的字符串表示，提取张量的ID和类型信息，并将其存储在类的属性中。
        输入张量存储在`self.inputs`字典中，参数张量存储在`self.param_id_to_name`字典中。
        参数:
        - state_dict (dict): 包含模型参数的字典，用于确定参数张量的数量。
        - torch_graph_str (str): PyTorch计算图的字符串表示，用于解析张量信息。
        返回值:
        - 无
        """
        # 检查计算图字符串是否以"graph"开头，确保字符串格式正确
        prefix_str = "graph"

        '''
        prefix_str = "graph": 定义了一个字符串变量 prefix_str，其值为 "graph"。
        torch_graph_str.startswith(prefix_str): 这是一个字符串方法调用。
        startswith() 方法检查 torch_graph_str 字符串是否以 prefix_str（即 "graph"）开头。
        如果 torch_graph_str 的起始部分是 "graph"，则该表达式返回 True；否则返回 False。
        '''
        assert torch_graph_str.startswith(prefix_str), "Invalid graph str to be parsed"

        # 这行代码调用了一个名为 _get_str_inside_parenthesis 的辅助函数，其目的是从 PyTorch 计算图的字符串表示 (torch_graph_str) 中提取出定义张量信息的部分。
        # 获取第一个顶级括号内的内容字符串，即graph之后第一对括号的字符串
        tensors_str = _get_str_inside_parenthesis(torch_graph_str, prefix_str = prefix_str)

        '''
        # tensors_str是一个字符串，这个字符串中的每个张量定义通常以百分号 % 开头，
        # split('%') 方法会以字符 % 作为分隔符，将 tensors_str 字符串分割成一个子字符串列表。
        # 选取从索引1开始到列表末尾的所有元素，并去除空白字符，
        # 遍历tensors_str.split('%')产生的列表中的每一个子字符串s。
        # s.strip() 方法会移除每个子字符串 s 两端的空白字符（包括空格、制表符、换行符等）。
        '''
        tensors_str_list = [s.strip() for s in tensors_str.split('%')][1:]

        # 初始化参数字典，用于存储参数ID到参数名称的映射关系
        self.param_id_to_name = dict()

        # 模型的实际输入张量，总张量数减去参数张量数
        # 这里的 state_dict 是模型的参数字典，len(state_dict) 返回参数张量的数量。
        # tensors_str_list 是从计算图字符串中解析出的所有张量信息列表，len(tensors_str_list) 返回所有张量的数量。
        num_inputs = len(tensors_str_list) - len(state_dict)
        print(f"--- num_inputs (模型实际输入张量数量): {num_inputs} ---")
        # 当前已处理的输入张量数量
        cur_input = 0
        # 当前已处理的参数张量数量
        cur_param = 0

        # 遍历每个张量信息字符串，解析并分类存储
        for i, tensor_str in enumerate(tensors_str_list):
            # i 是当前遍历到的张量在 tensors_str_list 中的索引，tensor_str 是当前张量信息的字符串，例如 "input.1 : Float(1, 3, 224, 224)" 或 "1 : Float(64, 3, 7, 7)"
            # 将张量信息字符串按':'分割，并去除每个部分的空白字符
            # 例如，"input.1 : Float(1, 3, 224, 224)" 会被分割成 ["input.1", "Float(1, 3, 224, 224)"]
            tensor_str_split = [s.strip() for s in tensor_str.split(":")]
            
            # 取tensor的ID，分割后的第一个元素是张量的ID（在计算图字符串中的表示，如 "input.1" 或 "1"）
            tensor_id = tensor_str_split[0] 
            
            # 根据当前张量的索引 i 和先前计算的 num_inputs 来判断张量类型，假设在计算图字符串的头部，输入张量总是列在参数张量之前
            # 前 num_inputs 个张量被认为是模型的实际输入张量，其余的张量被认为是模型的参数张量
            tensor_type = "input" if i < num_inputs or tensor_id.startswith('input.') else "params"
            
            # 如果判断当前张量是输入张量
            if tensor_type == "input":
                # 将输入张量的信息存储到 self.inputs 字典中
                # 键的格式为 'node-i'，其中 i 是该输入张量在 tensors_str_list 中的原始索引
                # 值是一个元组，包含:
                #   - i: 输入张量的原始索引
                #   - tensor_id: 输入张量在图字符串中的ID
                #   - tensor_str_split: 分割后的张量信息字符串列表 (包含ID和类型/形状)
                self.inputs['node-' + str(i)] = (i, tensor_id, tensor_str_split)
                # 输入张量计数器加1
                cur_input += 1
            # 如果判断当前张量是参数张量
            elif tensor_type == "params":
                # 将参数张量的ID（转换为整数，因为图中的参数ID通常是数字）映射到其真实的参数名称
                # self.param_names 是一个预先收集好的模型参数名称列表 (通常按 state_dict 的顺序)
                # cur_param 用作索引，从 self.param_names 中按顺序取出参数名
                # 这样就建立了计算图中参数ID（如 %1, %2）与模型中实际参数名（如 'conv1.weight', 'fc.bias'）的对应关系
                if tensor_id.isdigit():
                    # 如果 tensor_id 不是纯数字，可能是一个特殊的图输入/状态
                    # print(f"  [Input/State Tensor] Index {i}: Non-numeric Graph ID '{tensor_id}' classified as 'param'.")
                    self.param_id_to_name[int(tensor_id)] = self.param_names[cur_param]
                else:
                    self.param_id_to_name[int(cur_param + 1)] = self.param_names[cur_param]
                # 参数张量计数器加1
                cur_param += 1         
    
    def build_dot(self, vertical=False, by_node_groups=True, display_params=True, display_flops=True):
        """
        构建并返回一个 GraphViz 的 Digraph 对象，用于可视化当前的图结构。
        该方法允许用户通过多种参数详细控制生成图形的视觉表现和信息密度。
        主要功能包括：
        - 将图的节点和边转换为 DOT 语言描述。
        - 支持垂直或水平布局。
        - 可选择是否按节点组（NodeGroup）对节点进行组织和着色，便于理解模块化结构。
        - 可选择是否在节点标签中展示参数的名称和形状。
        - 可选择是否在节点标签中展示其 FLOPs（浮点运算次数）信息，有助于性能分析。
        参数:
            vertical (bool, 默认为 False): 
                如果为 True，图形将采用从上到下的垂直布局。
                如果为 False，图形将采用从左到右的水平布局。
            by_node_groups (bool, 默认为 True): 
                如果为 True，节点将根据其所属的节点组进行分组和着色。
                这有助于识别和区分图中的不同功能单元或可剪枝单元。
            display_params (bool, 默认为 True): 
                如果为 True，每个节点的标签中将包含其关联参数的名称和形状信息。
            display_flops (bool, 默认为 True): 
                如果为 True，每个节点的标签中将包含其 FLOPs（浮点运算次数）信息，
                通常以占模型总 FLOPs 的百分比形式显示。
        返回:
            graphviz.Digraph: 一个 Digraph 对象，可以直接用于渲染（如保存为图片文件）或进一步操作。
        """
        from graphviz import Digraph    # graphviz 用于创建和渲染 DOT 图
        import random                   # random 用于生成随机颜色

        # 初始化一个字典，用于存储 FLOPs（浮点运算次数）的细分数据。
        flops_break_down = dict()
        # 如果 display_flops 参数为 True，则计算模型的 FLOPs。
        if display_flops:
            # 调用 compute_flops 方法计算 FLOPs，结果以百万为单位。
            flops_break_down = self.compute_flops(in_million=True)

        # 创建一个 Digraph 对象，这是 GraphViz 图的核心。
        dot = Digraph()
        
        # 设置图的全局属性，如背景色、边框色、字体、边距、布局方向等。
        # 这些属性来自 self.theme 字典，它定义了图的视觉样式。
        dot.attr("graph", 
                bgcolor=self.theme["background_color"], # 背景颜色
                color=self.theme["outline_color"],      # 图形轮廓颜色
                fontsize=self.theme["font_size"],       # 字体大小
                fontcolor=self.theme["font_color"],     # 字体颜色
                fontname=self.theme["font_name"],       # 字体名称
                margin=self.theme["margin"],            # 图形外边距
                rankdir="TB" if vertical else "LR",     # 图的布局方向，"TB" 表示从上到下（垂直），"LR" 表示从左到右（水平）
                pad=self.theme["padding"])              # 图形内边距

        # 设置边的全局属性，如样式、颜色、字体等。
        dot.attr("edge", style="solid",                 # 边的样式，"solid" 表示实线
                color=self.theme["outline_color"],      # 边的颜色
                fontsize=self.theme["font_size"],       # 边标签的字体大小
                fontcolor=self.theme["font_color"],     # 边标签的字体颜色
                fontname=self.theme["font_name"])       # 边标签的字体名称
        
        # 构建 GraphViz Digraph（有向图），判断是否需要按节点组（node_groups）来绘制图。
        # 如果没有节点组，或者 by_node_groups 参数为 False，则直接绘制所有节点。
        if len(self.node_groups) == 0 or not by_node_groups:
            # 遍历图中的所有节点。
            for node in self.nodes.values():
                # 特殊处理 dummy_input 节点（通常表示模型的输入）。
                if node.id == "dummy_input":
                    # 设置 dummy_input 节点的属性，如形状（椭圆）、填充样式、颜色等。
                    dot.attr("node", shape="ellipse",               # 设置节点的形状（椭圆）
                            style="filled", margin="0,0",           # 填充样式（填充）和边框宽度（0）
                            fillcolor=self.theme["fill_color"],     # 设置节点的填充颜色
                            color=self.theme["outline_color"],      # 设置节点的边框颜色
                            fontsize=self.theme["font_size"],       # 设置节点标签的字体大小
                            fontname=self.theme["font_name"])       # 设置节点标签的字体名称
                    # 创建节点的标签，使用 HTML 表格格式，只显示节点 ID。
                    # 这行代码使用字符串的 format 方法，将 node.id（节点的唯一标识符）插入到 HTML 表格的单元格 (<td>) 中。
                    # cellpadding='6' 是 HTML 表格单元格的属性，表示单元格内容与其边框之间的内边距为6个单位。
                    # <tr>...</tr> 定义了表格中的一行。
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.id)
                    # 这行代码将上一行生成的单个表格行（包含节点ID）包装在一个完整的 HTML 表格结构中。
                    # "<<table ... > ... </table>>" 是 Graphviz 用来渲染复杂节点标签的特殊语法，允许使用 HTML 子集。
                    # border='0' 表示表格本身没有边框。
                    # cellborder='0' 表示表格单元格之间没有边框。
                    # cellpadding='0' 表示表格单元格的内边距为0（注意：这里是表格级别的内边距，而上一行是单元格级别的）。
                    # 最终，label 变量存储的是一个完整的、可以被 Graphviz 解析的 HTML 字符串，用于显示节点标签。
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    # 将节点添加到图中。
                    dot.node(str(node.id), label)
                # 特殊处理 dummy_output 节点（通常表示模型的输出）。
                elif node.id == "dummy_output":
                    # 设置 dummy_output 节点的属性，如形状（双八边形）、填充样式、颜色等。
                    dot.attr("node", shape="doubleoctagon",     # 设置节点的形状（双八边形）
                            style="filled", margin="0,0",       # 填充样式（填充）和边框宽度（0）
                            fillcolor=self.theme["fill_color"], # 设置节点的填充颜色
                            color=self.theme["outline_color"],  # 设置节点的边框颜色
                            fontsize=self.theme["font_size"],   # 设置节点标签的字体大小
                            fontname=self.theme["font_name"])   # 设置节点标签的字体名称
                    # 创建节点的标签，使用 HTML 表格格式，只显示节点 ID。
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.id)
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    # 将节点添加到图中。
                    dot.node(str(node.id), label)
                # 处理普通节点。
                else:
                    # 设置普通节点的属性，如形状（矩形）、填充样式、颜色等。
                    dot.attr("node", shape="box",               # 设置节点的形状（矩形）
                            style="filled", margin="0,0",       # 填充样式（填充）和边框宽度（0）
                            fillcolor=self.theme["fill_color"], # 设置节点的填充颜色
                            color=self.theme["outline_color"],  # 设置节点的边框颜色
                            fontsize=self.theme["font_size"],   # 设置节点标签的字体大小
                            fontcolor=self.theme["font_color"], # 设置节点标签的字体颜色
                            fontname=self.theme["font_name"])   # 设置节点标签的字体名称
                    # 创建节点的标签，显示节点的标题 (node.title)。
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.title)
                    # 如果节点有 ID，则在标签中额外显示节点 ID。
                    if node.id:
                        label += "<tr><td>{}</td></tr>".format(node.id)
                    # 将标签包装在 HTML 表格中。
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    # 将节点添加到图中。
                    dot.node(str(node.id), label)
        # 如果需要按节点组绘制图。
        else:
            # 创建一个字典，用于存储每个节点的颜色列表（一个节点可能属于多个组，颜色会叠加）。
            node_colors = dict()
            
            for node in self.nodes.values():
                # 遍历所有的节点
                node_colors[node.id] = list()   # 初始化每个节点的颜色列表为空。

            # 创建集合，用于存储属于可剪枝节点组和辅助节点组的节点 ID。
            nodes_in_prunable_node_groups = set()
            nodes_in_auxiliary_node_groups = set()
            # 遍历所有的节点组。
            for node_group in self.node_groups.values():
                # 为每个节点组生成一个随机颜色。
                random_number = random.randint(0,16777215) # 生成一个 0 到 2^24-1 之间的随机整数
                hex_number = str(hex(random_number))       # 将整数转换为十六进制字符串 (例如 "0xffa1b2")
                color ='#'+ hex_number[2:]                 # 提取十六进制颜色码 (例如 "#ffa1b2")
                # 获取节点组是否可剪枝和是否为辅助组的标志。
                is_prunable = node_group.is_prunable
                is_auxiliary = node_group.is_auxiliary
                # 遍历节点组中的每个节点。
                for node in node_group:
                    # 将当前节点组的颜色添加到该节点的颜色列表中。
                    node_colors[node.id].append(color)
                    # 如果节点是可剪枝的，则将节点 ID 添加到可剪枝节点集合中。
                    if is_prunable:
                        nodes_in_prunable_node_groups.add(node.id)
                    # 如果节点是辅助的，则将节点 ID 添加到辅助节点集合中。
                    if is_auxiliary:
                        nodes_in_auxiliary_node_groups.add(node.id)
                
            # 为那些不属于任何已着色节点组的节点设置默认填充色。
            for node_id in node_colors:
                if len(node_colors[node_id]) == 0:
                    node_colors[node_id] = self.theme["fill_color"]

            # 遍历图中的所有节点，并根据其属性和所属组来设置样式和标签。
            for node in self.nodes.values():
                # 特殊处理 dummy_input 节点。
                if node.id == "dummy_input":
                    dot.attr("node", shape="ellipse",           # 设置节点的形状（圆形）
                            style="filled", margin="0,0",       #  填充样式（填充）和边框宽度（0）
                            fillcolor=self.theme["fill_color"], # 设置节点的填充颜色
                            color=self.theme["outline_color"],  # 设置节点的边框颜色
                            fontsize=self.theme["font_size"],   # 设置节点标签的字体大小
                            fontcolor=self.theme["font_color"], #  设置节点标签的字体颜色
                            fontname=self.theme["font_name"])   #  设置节点标签的字体名称
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.id)                      # 创建节点的标签，使用 HTML 表格格式，只显示节点 ID。
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"  # 将标签包装在 HTML 表格中。
                    dot.node(str(node.id), label)                                                       # 将节点添加到图中。
                # 特殊处理 dummy_output 节点。
                elif node.id == "dummy_output":
                    dot.attr("node", shape="doubleoctagon",     # 设置节点的形状（双八边形）
                            style="filled", margin="0,0",       # 填充样式（填充）和边框宽度（0）
                            fillcolor=self.theme["fill_color"], # 设置节点的填充颜色
                            color=self.theme["outline_color"],  # 设置节点的边框颜色
                            fontsize=self.theme["font_size"],   # 设置节点标签的字体大小
                            fontcolor=self.theme["font_color"], # 设置节点标签的字体颜色
                            fontname=self.theme["font_name"])   # 设置节点标签的字体名称
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.id)                      # 创建节点的标签，使用 HTML 表格格式，只显示节点 ID。
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"  # 将标签包装在 HTML 表格中。
                    dot.node(str(node.id), label)                                                       # 将节点添加到图中。
                # 处理普通节点。
                else:
                    # 获取节点的颜色。如果节点属于多个组，颜色会以冒号分隔的形式组合（GraphViz支持渐变色）。
                    color = ":".join(node_colors[node.id])
                    # 如果节点没有参数。
                    if len(node.param_names) == 0:
                        # 设置节点的形状：如果不在辅助组中则为矩形（box），否则为椭圆（ellipse）。
                        # 设置节点的样式：如果属于可剪枝组则为填充（filled），否则为虚线（dashed）。
                        # 设置填充色、边框色、字体颜色等。
                        dot.attr("node", shape="box" if node.id not in nodes_in_auxiliary_node_groups else "ellipse",   #  节点的形状
                                style="filled" if node.id in nodes_in_prunable_node_groups else "dashed",               # 节点的样式
                                margin="0,0",       # 边框宽度
                                fillcolor=color,    # 节点的填充色
                                # 边框颜色：如果节点不在可剪枝组中，则使用节点自身的颜色，否则使用主题定义的轮廓色
                                color=color if node.id not in nodes_in_prunable_node_groups else self.theme["outline_color"],
                                fontsize=self.theme["font_size"],   # 节点标签的字体大小
                                # 字体颜色：如果节点不在可剪枝组中，则使用节点自身的颜色（可能与背景色相同，导致看不见），否则使用白色
                                fontcolor=color if node.id not in nodes_in_prunable_node_groups else "#FFFFFF",
                                fontname=self.theme["font_name"])   # 节点标签的字体名称
                    # 如果节点有参数。
                    elif len(node.param_names) > 0:
                        # 设置节点的形状、样式、颜色等，逻辑与无参数节点类似。
                        dot.attr("node", shape="box" if node.id not in nodes_in_auxiliary_node_groups else "ellipse", 
                                style="filled" if node.id in nodes_in_prunable_node_groups else "dashed", 
                                margin="0,0",
                                fillcolor=color,
                                color=color if node.id not in nodes_in_prunable_node_groups else self.theme["outline_color"],
                                fontsize=self.theme["font_size"],
                                fontcolor=self.theme["font_color"], # 有参数时，字体颜色统一使用主题定义的颜色
                                fontname=self.theme["font_name"])                        

                    # 创建节点的标签，首先显示节点标题。
                    label = "<tr><td cellpadding='6'>{}</td></tr>".format(node.title)
                    # 如果节点有 ID，则显示节点 ID。
                    if node.id:
                        label += "<tr><td>{}</td></tr>".format(node.id)
                    # 如果节点有参数且 display_params 为 True，则显示参数名称和形状。
                    if len(node.param_names) > 0 and display_params:
                        for p_name in node.param_names:
                            # 参数形状从 self.params_grad 或 self.params_no_grad 中获取。
                            label += "<tr><td>{}-{}</td></tr>".format(p_name, self.params_grad[p_name].shape if p_name in self.params_grad else self.params_no_grad[p_name].shape)
                    # 如果 display_flops 为 True，则显示该节点的 FLOPs 占总 FLOPs 的百分比。
                    if display_flops:
                        try:
                            # 如果 flops_break_down['total'] 为 0，则显示 "N/A (Total FLOPs is 0)"。
                            # 否则，计算该节点的 FLOPs 占总 FLOPs 的百分比。
                            if flops_break_down['total'] == 0:
                                label += "<tr><td>FLOPs- N/A (Total FLOPs is 0)</td></tr>"
                            else:
                                # 计算该节点的 FLOPs 占总 FLOPs 的百分比。
                                flops_percentage = flops_break_down['by_nodes'][node.id] / flops_break_down['total']
                                label += "<tr><td>FLOPs-{:.4f}</td></tr>".format(flops_percentage)
                        except ZeroDivisionError:
                            # 如果计算 FLOPs 时发生除以零错误，显示 "FLOPs- -1 (Division by zero)"。
                            label += "<tr><td>FLOPs- -1 (Division by zero)</td></tr>"
                        except KeyError:
                            # 如果节点 ID 不在 flops_break_down['by_nodes'] 中，显示 "FLOPs- N/A (KeyError)"。
                            label += "<tr><td>FLOPs- N/A (KeyError)</td></tr>"
                    # 将标签包装在 HTML 表格中。
                    label = "<<table border='0' cellborder='0' cellpadding='0'>" + label + "</table>>"
                    # 将节点添加到图中。
                    dot.node(str(node.id), label)

        # 遍历图中的所有边，并将它们添加到 Digraph 对象中。
        for a, b, label in self.edges: # a 是起始节点ID, b 是结束节点ID, label 是边的标签
            # 如果边的标签是列表或元组（通常表示张量的形状），则将其转换为 "x" 分隔的字符串。
            if isinstance(label, (list, tuple)):
                label = "x".join([str(l or "?") for l in label]) # "l or ?" 表示如果 l 为 None 或空，则显示 "?"
            # 将边添加到图中。
            dot.edge(str(a), str(b), label)
        return dot    

    def visited_dict(self):
        visited = dict()
        for node in self.nodes.values():
            visited[node.id] = False
        return visited

    def random_set_zero_groups(self, target_group_sparsity=None, num_group_divisible=1):
        """
        随机选择模型中可剪枝的参数组，并将这些组的部分权重设置为零。
        该函数主要用于在训练或推理过程中对模型进行结构化剪枝，模拟某些通道/头被移除的效果。
        参数:
            target_group_sparsity (float or None): 
                - 指定每个参数组中应被置零的比例（0.0 到 1.0之间）。
                - 如果为 None，则随机生成一个比例。
            num_group_divisible (int): 
                - 控制实际置零的组数必须是该值的倍数，默认为 2。
                - 用于保证剪枝后的结构仍满足硬件加速要求（如多头注意力等）。
        返回值:
            无显式返回值。该方法直接修改模型参数的数据（in-place操作）。
        """
        # 获取所有可训练的参数组信息，每组包含参数名称、张量、变换方式等
        param_groups = self.get_param_groups()
        
        # 遍历所有参数组
        for param_group in param_groups:
            # 跳过不可剪枝或辅助性的参数组
            if not param_group['is_prunable'] or param_group['is_auxiliary']:
                continue
            # 确保目标稀疏度合法（0 <= sparsity < 1），若未指定则随机生成
            assert target_group_sparsity is None or (target_group_sparsity >= 0 and target_group_sparsity < 1.0)
            curr_group_sparsity = np.random.random() if target_group_sparsity is None else target_group_sparsity
            # 计算当前参数组中总共有多少个分组（num_groups）
            num_groups = param_group['num_groups']
            # 根据稀疏度和num_group_divisible计算需要置零的组数量
            # 确保至少为0，最多不超过 num_groups - 1（不能全置零）
            num_zero_groups = max(min(int(curr_group_sparsity * num_groups) // num_group_divisible * num_group_divisible, num_groups - 1), 0)
            # 随机选择要置零的组索引（从0到num_groups-1中选num_zero_groups个）
            zero_group_idxes = np.random.choice(list(range(0, num_groups - 1)), num_zero_groups, replace=False)
            zero_group_idxes.sort()  # 排序以保持一致性

            # 如果该参数组没有实际参数，跳过处理
            if len(param_group['params']) == 0:
                continue

            # 遍历该参数组中的每个参数（包括其名称、数据张量和变换方式）
            for (p_name, param, p_transform) in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
                # 跳过LoRA相关的不可剪枝参数
                if 'lora_A' in p_name or 'lora_embedding_A' in p_name:
                    continue

                # 根据不同的张量变换方式对参数进行置零操作
                if p_transform == TensorTransform.TRANSPOSE and len(param.data.shape) > 1:
                    # 对转置张量：按列（第1维）置零
                    param.data[:, zero_group_idxes, ...] = 0.0
                elif p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    # 多头注意力（head dim模式）：每个 head 的对应位置都置零
                    multi_head_zero_group_idxes = zero_group_idxes.tolist()
                    for h in range(1, param_group['num_heads']):
                        # 每个 head 的偏移量为 head_dim * h
                        multi_head_zero_group_idxes.extend([i + param_group['head_dim'] * h for i in zero_group_idxes.tolist()])
                    param.data[multi_head_zero_group_idxes] = 0.0
                elif p_transform == TensorTransform.MULTIHEAD_NUMHEAD or p_transform == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
                    # 多头注意力（num head模式）：每个 head 内部多个位置置零
                    multi_head_zero_group_idxes = list()
                    for i in zero_group_idxes.tolist():
                        for h in range(param_group['head_dim']):
                            # 每个 head 中的位置为 i * head_dim + h
                            multi_head_zero_group_idxes.append(h + i * param_group['head_dim'])
                    param.data[multi_head_zero_group_idxes] = 0.0
                elif isinstance(p_transform, list):
                    refined_zero_idxes = [i for i in zero_group_idxes]
                    for (p_transform_type, p_transform_config) in reversed(p_transform):
                        if p_transform_type == TensorTransform.MULTIHEAD_HEADDIM:
                            head_dim = p_transform_config['head_dim']
                            num_heads = p_transform_config['num_heads']
                            refined_zero_idxes = index_transformation(refined_zero_idxes, p_transform_type, num_heads=num_heads, head_dim=head_dim)
                        elif p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD or p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
                            head_dim = p_transform_config['head_dim'] 
                            refined_zero_idxes = index_transformation(refined_zero_idxes, p_transform_type, head_dim=head_dim)
                    param.data[refined_zero_idxes] = 0.0
                else:
                    # 默认情况：直接对指定组索引置零
                    param.data[zero_group_idxes] = 0.0

            # 处理与当前参数组关联的辅助节点组（auxiliary node groups）
            for ng_id, offset in param_group['auxiliary_ngs']:
                aux_pg = self.node_groups[ng_id].get_param_groups()
                for aux_p in aux_pg['params']:
                    # 在辅助组中对应位置也置零（考虑偏移量）
                    aux_p.data[offset+zero_group_idxes, ...] = 0.0
        
    def set_pruning_redundant_idxes(self):
        """
        遍历所有节点组，设置剪枝过程中冗余的索引。
        该方法首先处理所有可剪枝且非辅助类型的节点组，随后处理辅助类型的节点组，调用各自的方法来确定在模型剪枝过程中应被标记为冗余的通道或节点索引。
        参数:
            无  
        返回值:
            无
        """
        # 处理所有可剪枝且非辅助类型的节点组，设置其冗余索引
        for node_group in self.node_groups.values():
            if node_group.is_prunable and not node_group.is_auxiliary:
                node_group.set_pruning_redundant_idxes()
        # 处理所有辅助类型的节点组，设置其冗余索引
        for node_group in self.node_groups.values():
            if node_group.is_auxiliary:
                node_group.set_pruning_redundant_idxes()
        
    def skip_operators(self, operators=list()):
        '''
        函数功能：
            将包含指定类型操作符（operators）的节点组标记为不可剪枝（unprunable）。
        参数说明：
            operators (list): 一个字符串列表，每个字符串表示需要跳过（即不进行剪枝）的操作符类型名称。
        返回值：
            None: 该函数没有显式返回值，而是通过修改内部节点组的 `is_prunable` 属性来影响对象状态。
        '''
        # 遍历图中的所有节点组
        for node_group in self.node_groups.values():
            # 如果当前节点组无参数或已被标记为不可剪枝，则跳过处理
            if len(node_group.param_names) == 0 or not node_group.is_prunable:
                continue
            # 处理组合操作类型的节点组（NodeGroupComposedOp）
            if type(node_group).__name__ == 'NodeGroupComposedOp':
                # 如果该节点组的操作类型在目标操作符列表中，将其标记为不可剪枝
                if node_group.op._type in operators:
                    node_group.is_prunable = False
            # 处理普通操作类型的节点组（NodeGroup）
            elif type(node_group).__name__ == 'NodeGroup':
                # 遍历该节点组内的所有节点
                for node in node_group:
                    # 如果当前节点无参数或无有效操作符，跳过处理
                    if len(node.param_names) == 0 or not node.op:
                        continue
                    # 如果当前节点的操作类型在目标操作符列表中，将整个节点组标记为不可剪枝，并跳出循环
                    if node.op._type in operators:
                        node_group.is_prunable = False
                        break
    
    def set_trainable(self):
        """
        根据模型参数的梯度状态更新各个节点组的可训练（is_trainable）和可剪枝（is_prunable）属性。
        该方法首先调用 `set_param_grad_no_grad` 方法，根据模型参数的 requires_grad 状态将参数分为需要梯度和不需要梯度两类。
        然后遍历所有节点组，判断每个节点组是否包含可训练参数，并据此设置其 is_trainable 和 is_prunable 属性。
        参数：
            无显式参数。使用类成员变量：
            - self._model: PyTorch 模型对象，用于获取参数的梯度状态。
            - self.node_groups: 节点组字典，存储图中所有的节点组。
            - self.params_grad: 包含可训练参数（requires_grad=True）的字典。
        返回值：
            无返回值。
        """
        # 更新 params_grad 和 params_no_grad 字典，分别记录模型中需要梯度和不需要梯度的参数
        self.set_param_grad_no_grad(self._model)
        # 遍历所有节点组，确定每个节点组是否可训练
        for node_group in self.node_groups.values():
            # 默认将当前节点组设为不可训练
            node_group.is_trainable = False

            # 如果当前节点组没有参数，则标记为不可训练且不可剪枝
            if len(node_group.param_names) == 0:
                node_group.is_tranable = False  
                node_group.is_prunable = False
                continue
            all_param_no_grad = True  # 标记当前节点组的所有参数都不需要梯度
            # 遍历当前节点组中的所有参数名
            for param_name in node_group.param_names:
                # 如果至少有一个参数在可训练参数列表中，则标记当前节点组为可训练
                if param_name in self.params_grad:
                    node_group.is_trainable = True
                    all_param_no_grad = False
                    break  # 只要发现一个可训练参数，即可停止检查
            # 如果当前节点组中所有参数都不需要梯度，则标记为不可剪枝
            if all_param_no_grad:
                node_group.is_prunable = False
    

    def set_param_grad_no_grad(self, model):
            """
            根据模型参数的 requires_grad 属性，将模型参数分为 需要梯度（可训练）和 不需要梯度（不可训练或缓冲区）两类。
            同时，收集模型状态字典中的所有键名。
            参数:
                model (torch.nn.Module): 需要分析的 PyTorch 模型对象。
            功能:
                - 初始化 self.params_grad 和 self.params_no_grad 两个字典。
                - 遍历 model.named_parameters()，根据 param.requires_grad 将参数及其名称分别存入上述两个字典。
                - 遍历 model.state_dict()，将所有键名（包括参数和缓冲区）存入 self.param_names 列表。
                - 对于 state_dict 中存在但不在 self.params_grad 中的项（通常是 requires_grad=False 的参数或缓冲区），
                  将其名称和值存入 self.params_no_grad 字典。
            """
            # 初始化用于存储需要梯度（可训练）参数的字典
            self.params_grad = dict()
            # 初始化用于存储不需要梯度（不可训练或缓冲区）参数/状态的字典
            self.params_no_grad = dict()
    
            
            # print("输出模型所有已命名的参数: ")
            # print(f"{'Parameter name':} {'Shape':}")    # 输出格式
            # 遍历模型的所有命名参数（通常是权重、偏置等）
            for name, param in model.named_parameters():
                # 查看模型的所有参数
                # print(f"Parameter name: {name}, Shape: {param.shape}")
                # 检查参数是否需要计算梯度
                if param.requires_grad:
                    # 如果需要梯度，则将其名称和参数对象存入 params_grad 字典
                    self.params_grad[name] = param
                else:
                    # 如果不需要梯度，则将其名称和参数对象存入 params_no_grad 字典
                    self.params_no_grad[name] = param

            # 遍历模型状态字典中的所有键名（包括参数和缓冲区，如BN层的running_mean/running_var）
            for name in model.state_dict():
                # 将状态字典中的键名添加到 param_names 列表中，用于记录模型的所有状态项
                self.param_names.append(name)
                # 检查当前键名在不在需要梯度的参数字典中，这通常意味着它是 requires_grad=False 的参数，或者是缓冲区（buffer）
                if name not in self.params_grad:
                    # 将这些不需要梯度的参数或缓冲区的名称和值（从state_dict获取）存入 params_no_grad 字典
                    self.params_no_grad[name] = model.state_dict()[name]            
            
    def get_param_groups(self):
        """
        获取所有可训练的参数组信息，用于优化器分组或结构化剪枝。
        该方法会遍历所有节点组，收集可训练的参数组，并处理辅助节点组与依赖节点组的关系。
        最后会移除完全不可训练的参数组，并按ID排序返回。
        返回值:
            list: 包含所有有效参数组信息的列表，每个元素为一个字典，包含参数名、参数张量、变换方式、组数量、是否可剪枝、辅助组信息等。
        """
        param_groups = dict()  # 初始化一个字典，用于存储节点组ID到参数组信息的映射
    
        # 第一遍遍历：收集所有可训练节点组的参数信息
        for node_group in self.node_groups.values():
            # 遍历图中的所有节点组
            if node_group.is_trainable:
                # 如果该节点组是可训练的（包含至少一个requires_grad=True的参数）
                ng_param_group = node_group.get_param_groups()  # 调用节点组自身的get_param_groups方法获取详细参数信息
                if len(ng_param_group['params']) > 0:
                    # 如果该参数组确实包含参数（避免空组）
                    param_groups[node_group.id] = ng_param_group  # 将参数组信息存入字典，key为节点组ID
    
        # 第二遍遍历：处理辅助节点组（auxiliary node groups）与其依赖的主节点组的关系
        for node_group in self.node_groups.values():
            if hasattr(node_group, 'auxilary_node_groups'):
                depend_ng_pg = param_groups[node_group.id]
                for aux_ng, offset in node_group.auxilary_node_groups:
                    if aux_ng.is_auxiliary and aux_ng.is_trainable:
                        depend_ng_pg['auxiliary_ngs'].append((aux_ng.id, offset))

        untrainable_param_group_ids = set()
        for param_group in param_groups.values():
            # 遍历已收集的参数组
            if len(param_group['auxiliary_ngs']) > 0:
                # 如果该参数组有关联的辅助组，暂时不判断其可训练性（依赖主组）
                continue
            all_params_no_req_grad = True  # 假设该组所有参数都不可训练
            for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
                # 遍历参数组中的每个参数
                if param.requires_grad:
                    # 如果发现至少一个参数需要梯度
                    all_params_no_req_grad = False  # 标记为包含可训练参数
            if all_params_no_req_grad:
                # 如果遍历完所有参数，发现都不需要梯度
                untrainable_param_group_ids.add(param_group['id'])  # 将该参数组ID加入待移除集合
    
        # 移除标记为完全不可训练的参数组
        for remove_id in untrainable_param_group_ids:
            del param_groups[remove_id]
        param_groups = dict(sorted(param_groups.items(), key=lambda kv:(kv[0], kv[1]))) # 按节点组ID对参数组进行排序，保证每次获取的顺序一致
        
        seen_param_ids = set()
        for group_id, group in list(param_groups.items()):
            params = list(group.get('params', []))
            if not params:
                del param_groups[group_id]
                continue
            names = list(group.get('p_names', [''] * len(params)))

            keep_indices = []
            unique_params = []
            unique_names = []
            for idx, (name, param) in enumerate(zip(names, params)):
                if param is None:
                    continue
                pid = id(param)
                if pid in seen_param_ids:
                    continue
                seen_param_ids.add(pid)
                keep_indices.append(idx)
                unique_params.append(param)
                unique_names.append(name)

            if not keep_indices:
                del param_groups[group_id]
                continue

            filtered_group = dict(group)
            filtered_group['params'] = unique_params
            filtered_group['p_names'] = unique_names
            for key, value in group.items():
                if key in ('params', 'p_names'):
                    continue
                if isinstance(value, list) and len(value) == len(params):
                    filtered_group[key] = [value[idx] for idx in keep_indices]
            param_groups[group_id] = filtered_group
        return param_groups.values()    # 返回排序后的参数组信息列表

    def get_node_groups_by_param_name(self, param_name=''):
        """
        根据给定的参数名称查找并返回包含该参数的所有节点组。
        参数:
            param_name (str): 需要查找的参数名称。默认为空字符串。
        返回:
            list: 包含所有指定参数名称的 NodeGroup 对象的列表。如果找不到，则返回空列表。
        用途:
            - 在剪枝或分析过程中，快速定位与特定参数（如某个层的权重或偏置）相关的节点组。
            - 便于对特定参数所在的结构单元进行操作或分析。
        """
        node_groups = list()  # 初始化一个空列表，用于存储找到的节点组
        # 遍历图中的所有节点组（存储在 self._graph.node_groups 字典的值中）
        # 注意：根据类结构，这里可能应该是 self.node_groups
        for node_group in self._graph.node_groups.values():
            # 检查当前节点组的 param_names 列表（包含该组所有参数的名称）
            # 是否包含传入的 param_name
            if param_name in node_group.param_names:
                # 如果包含，则将该节点组对象添加到结果列表中
                node_groups.append(node_group)
        # 返回包含所有匹配节点组的列表
        return node_groups

    def compute_flops(self, in_million=True, in_billion=False):
        """
        计算整个模型的 FLOPs（浮点运算次数），并按节点组和单个节点进行细分统计。
        参数:
            in_million (bool): 是否将 FLOPs 转换为百万单位（1e6）
            in_billion (bool): 是否将 FLOPs 转换为十亿单位（1e9）
        返回值:
            dict: 包含以下键的字典：
                - 'total': 模型总 FLOPs（已根据单位转换）
                - 'by_node_groups': 每个节点组的 FLOPs 总和
                - 'by_nodes': 每个节点的 FLOPs 明细
        用途说明:
            - 剪枝分析时评估计算量变化；
            - 可视化节点/节点组的计算占比；
            - 模型压缩、性能优化参考；
        """
        # 初始化 flops_break_down 字典，用于存储不同层级的 FLOPs 统计
        flops_break_down = dict()
        flops_break_down['total'] = 0                       # 存储模型总 FLOPs
        flops_break_down['by_node_groups'] = dict()         # 存储每个节点组的 FLOPs
        flops_break_down['by_nodes'] = dict()               # 存储每个节点的 FLOPs 明细
        
        # 遍历所有节点组，计算其内部所有节点的 FLOPs 并累加
        for node_group in self.node_groups.values():
            # 初始化当前节点组的 FLOPs 为 0
            flops_break_down['by_node_groups'][node_group.id] = 0
            # 遍历该节点组中的每一个节点
            for node in node_group:
                '''            
                # ---- START DEBUG PRINT ----
                print(f"DEBUG_FLOPs: Processing Node ID: {node.id}, OpName: {node.op_name}, OpType: {type(node.op)}")
                print(f"DEBUG_FLOPs: Node Input Shapes: {node.input_shape}")
                print(f"DEBUG_FLOPs: Node Output Shapes: {node.output_shape}")
                '''
                
                if not node.op or not hasattr(node.op, 'compute_flops'):
                    print(f"DEBUG_FLOPs: Node {node.id} has no op or op has no compute_flops method. Skipping.")
                    flops_break_down['by_nodes'][node.id] = 0
                    continue

                if node.input_shape is None:
                    print(f"CRITICAL_FLOPs_ERROR: node.input_shape is None for Node ID: {node.id} (OpName: {node.op_name}). This will cause a TypeError.")
                    flops_break_down['by_nodes'][node.id] = 0 # 记录为0，避免后续计算错误
                    # flops_break_down['total'] += 0 # 总数不变
                    continue # 跳过此节点的FLOPs计算
                elif not node.input_shape: # 检查 input_shape 是否为空列表/元组
                    print(f"CRITICAL_FLOPs_ERROR: node.input_shape is empty for Node ID: {node.id} (OpName: {node.op_name}). Cannot access input_shape[0].")
                    flops_break_down['by_nodes'][node.id] = 0
                    continue
                elif node.input_shape[0] is None:
                    print(f"CRITICAL_FLOPs_ERROR: node.input_shape[0] is None for Node ID: {node.id} (OpName: {node.op_name}). This is the direct cause of the TypeError.")
                    flops_break_down['by_nodes'][node.id] = 0
                    continue
                
                # 调用节点操作对象（op）的 compute_flops 方法，传入输入 shape
                cur_flops = node.op.compute_flops(node.input_shape[0])
                # 使用 _scale_value 工具函数将 FLOPs 转换为指定单位（百万或十亿）
                cur_flops = _scale_value(cur_flops, in_million, in_billion)
                # print(f"FLOPs - Node ID: {node.id}, OpName: {node.op_name}, Raw FLOPs: {cur_flops}")

                # 将当前节点的 FLOPs 累加到对应节点组中
                flops_break_down['by_node_groups'][node_group.id] += cur_flops
                # 将当前节点的 FLOPs 存入 by_nodes 字典，key为节点id
                flops_break_down['by_nodes'][node.id] = cur_flops
                # 将当前节点的 FLOPs 累加到模型总 FLOPs 中
                flops_break_down['total'] += cur_flops
        # 返回最终包含各层级FLOPs统计的字典
        return flops_break_down

    def compute_num_params(self, in_million=True, in_billion=False):
        """
        计算模型中所有参数的总数量，并按照指定单位进行缩放。
        参数:
            in_million (bool): 如果为 True，输出的参数数量将转换为百万单位（1e6）。默认值为 True。
            in_billion (bool): 如果为 True，输出的参数数量将转换为十亿单位（1e9）。默认值为 False。
        返回:
            float: 缩放后的参数总数。如果 both in_million 和 in_billion 为 True，则优先使用 in_billion 的单位进行缩放。
        """
        # 初始化 num_params 变量为 0，用于存储模型参数的总数量（以参数个数计算）
        num_params = 0
        # 遍历模型的所有命名参数（包括权重和偏置等）
        for _, param in self._model.named_parameters():
            """
            在每次迭代中：
            - '_' 是参数的名称（例如 "weight" 或 "bias"），我们在这里不需要处理它
            - 'param' 是一个 PyTorch 张量对象，代表某个通路的具体参数数据
            """
            # 将当前参数张量中元素的数量（numel）累加到 num_params 中
            num_params += param.numel()
            """
            示例：假设参数张量的形状是 (32, 64, 3, 3)，
                则 param.numel() = 32 * 64 * 3 * 3 = 18432，
                并将其加到总的 num_params 上。
            """
        # 调用 _scale_value 函数对参数总数进行单位缩放
        return _scale_value(num_params, in_million, in_billion)

    def cluster_node_groups(self, num_clusters=1):
        """
        将图中可剪枝且可训练的节点组进行聚类。
        参数:
            num_clusters (int): 希望将节点组聚成的簇的数量。默认为1。
        功能说明:
        - 如果 num_clusters 为 1，则所有符合条件（可剪枝且可训练）的节点组都被放入同一个簇（ID为0）。
        - 如果 num_clusters 大于 1，则使用 KMeans 算法对符合条件的节点组进行聚类。
          聚类的特征基于每个节点组的 'num_groups' 属性（可能代表通道数、头数等可分组的数量）。
        - 聚类结果存储在 self.node_group_clusters 字典中，key 为簇 ID，value 为该簇包含的 NodeGroup 对象列表。
        用途:
        - 在结构化剪枝中，可以将相似规模或类型的节点组划分到同一个簇，便于统一处理或应用不同的剪枝策略。
        - 可以用于分析模型结构，识别具有相似特征的计算单元组。
        """
        if num_clusters == 1:
            # 如果只需要一个簇
            self.node_group_clusters = dict()       # 初始化簇字典
            self.node_group_clusters[0] = list()    # 创建 ID 为 0 的簇
            for node_group in self.node_groups.values():
                # 遍历所有节点组
                if not node_group.is_prunable or not node_group.is_trainable:
                    # 如果节点组不可剪枝或不可训练，则跳过
                    continue
                # 将符合条件的节点组添加到簇 0 中
                self.node_group_clusters[0].append(node_group)
        else:
            # 如果需要多个簇，使用 KMeans 聚类
            from sklearn.cluster import KMeans  # 导入 KMeans 库
    
            node_group_ids = []     # 存储符合条件的节点组 ID
            node_group_sizes = []   # 存储用于聚类的节点组特征
            for node_group in self.node_groups.values():
                # 遍历所有节点组
                if not node_group.is_prunable or not node_group.is_trainable:
                    # 如果节点组不可剪枝或不可训练，则跳过
                    continue
                # 记录符合条件的节点组 ID
                node_group_ids.append(node_group.id)
                # 将节点组的数量和一个常数 1.0 作为聚类特征
                # num_groups 节点组中的组数量
                # 1.0 可能是为了构成二维特征，或者有其他特定含义
                node_group_sizes.append([node_group.num_groups, 1.0])
            # 将特征列表转换为 NumPy 数组，以便 KMeans 处理
            node_group_sizes = np.array(node_group_sizes)

            # 打印符合条件的节点组 ID 和特征
            print("\n--- 用于KMeans聚类的节点组ID和特征 (node_group_ids, node_group_sizes) ---")
            if not node_group_ids:
                print("  没有符合条件（可剪枝且可训练）的节点组用于聚类。")
            else:
                print(f"  共 {len(node_group_ids)} 个节点组参与聚类:")
                for i in range(len(node_group_ids)):
                    group_id = node_group_ids[i]
                    features = node_group_sizes[i]
                    original_group = self.node_groups.get(group_id)
                    num_groups_attr = original_group.num_groups if original_group and hasattr(original_group, 'num_groups') else "N/A"
                    print(f"    - 节点组 ID: {group_id:<20} | 原始 num_groups: {str(num_groups_attr):<5} | 用于聚类的特征: {features}")
            print("-" * 70)

    
            # 执行 KMeans 聚类
            # 1. KMeans(...): 这是从 scikit-learn 库的 cluster 模块中导入的 KMeans 类的实例化。
            #    KMeans 是一种非常流行的无监督聚类算法，它的目标是将 n 个观测值划分为 k 个簇，
            #    使得每个观测值都属于具有最近均值（簇质心或簇中心）的簇。
            # 2. n_clusters=num_clusters:
            #    - n_clusters: 这是 KMeans 类的一个关键参数，用于指定要形成的簇的数量（即 k 值）。
            #    - num_clusters: 这个变量是从 `cluster_node_groups` 方法的参数传递过来的，表示用户希望将节点组划分成多少个簇。
            # 3. random_state=0:
            #    - random_state: 这个参数用于控制 KMeans 算法中质心初始化的随机性。
            #      KMeans 算法的初始质心选择会影响最终的聚类结果。
            #      通过设置一个固定的整数（这里是 0），可以确保每次运行代码时，只要输入数据 (node_group_sizes) 和其他参数相同，KMeans 的初始化过程和最终的聚类结果都是可复现的。
            # 4. n_init="auto":
            #    - n_init: 这个参数指定了 KMeans 算法将使用不同质心种子运行的次数。
            #      算法会选择具有最低惯性（inertia，即簇内平方和）的那次运行结果作为最终结果。
            # 5. .fit(node_group_sizes):
            #    - .fit(): 这是 KMeans 对象的一个方法，用于在给定的数据上执行聚类计算。
            #    - node_group_sizes: 这是传递给 `fit` 方法的实际数据。
            #      它是一个 NumPy 数组（之前由 `node_group_sizes = np.array(node_group_sizes)` 创建），其中每一行代表一个节点组，每一列代表该节点组的一个特征。
            #      在这个特定的上下文中，`node_group_sizes` 的每一行是 `[node_group.num_groups, 1.0]`，意味着聚类是基于每个节点组的 `num_groups` 属性（以及一个常数特征）进行的。
            #    - 执行 `fit` 方法后，KMeans 对象 (kmeans) 内部会计算出簇的质心，并且每个输入数据点（即每个节点组）会被分配到一个簇。
            #      聚类结果（每个数据点所属的簇标签）可以通过 `kmeans.labels_` 属性访问。
            # 这行代码创建了一个 KMeans 聚类器实例，配置了簇的数量、随机种子以保证可复现性、
            # 以及初始化次数，然后使用 `node_group_sizes` 数据（代表可剪枝且可训练的节点组的特征）
            # 来训练这个聚类器，从而将这些节点组划分到指定的 `num_clusters` 个簇中。
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(node_group_sizes)
    
            self.node_group_clusters = dict()   # 初始化簇字典，用于存储聚类结果
            # 遍历聚类结果，将每个节点组分配到对应的簇
            for node_group_id, node_group_cluster_id in zip(node_group_ids, kmeans.labels_.tolist()):
                # node_group_id: 节点组的 ID
                # node_group_cluster_id: KMeans 分配的簇 ID
                if node_group_cluster_id not in self.node_group_clusters:
                    # 如果当前簇 ID 首次出现，则在字典中创建对应的空列表
                    self.node_group_clusters[node_group_cluster_id] = list()
                # 根据节点组 ID 获取节点组对象
                node_group = self.node_groups[node_group_id]
                # 将该节点组对象添加到其所属的簇列表中
                self.node_group_clusters[node_group_cluster_id].append(node_group)

    def get_node_groups_by_param_name(self, param_name=''):
        """
        根据给定的参数名称查找并返回包含该参数的所有节点组。
        参数:
            param_name (str): 需要查找的参数名称。默认为空字符串。
        返回:
            list: 包含所有包含指定参数名称的 NodeGroup 对象的列表。如果找不到，则返回空列表。
        用途:
            - 在剪枝或分析过程中，快速定位与特定参数（如某个层的权重或偏置）相关的节点组。
            - 便于对特定参数所在的结构单元进行操作或分析。
        """
        node_groups = list()  # 初始化一个空列表，用于存储找到的节点组
        # 遍历图中的所有节点组（存储在 self.node_groups 字典的值中）
        for node_group in self.node_groups.values():
            # 检查当前节点组的 param_names 列表（包含该组所有参数的名称）
            # 是否包含传入的 param_name
            if param_name in node_group.param_names:
                # 如果包含，则将该节点组对象添加到结果列表中
                node_groups.append(node_group)
        # 返回包含所有匹配节点组的列表
        return node_groups

    def get_nodes_by_param_name(self, param_name=''):
        """
        根据给定的参数名称查找并返回包含该参数的所有节点。
        参数:
            param_name (str): 需要查找的参数名称。默认为空字符串。
        返回:
            list: 包含所有包含指定参数名称的 Node 对象的列表。如果找不到，则返回空列表。
        用途:
            - 快速定位图中直接使用特定参数（如权重、偏置）的计算节点。
            - 便于分析参数在计算图中的具体使用位置。
        """
        nodes = list()  # 初始化一个空列表，用于存储找到的节点
        # 遍历图中的所有节点对象（存储在 self.nodes 字典的值中）
        for node in self.nodes.values():
            # 检查当前节点的 param_names 列表（包含该节点直接使用的所有参数的名称）
            # 是否包含传入的 param_name
            if param_name in node.param_names:
                # 如果包含，则将该节点对象添加到结果列表中
                nodes.append(node)
        # 返回包含所有匹配节点的列表
        return nodes
    