from abc import ABC, abstractclassmethod                                            						# 导入 ABC 和 abstractclassmethod 用于定义抽象基类和抽象方法
import torch                                                                        						# 导入 torch库，用于张量操作和神经网络
from only_train_once.transform import tensor_transformation, TensorTransform, index_transformation        	# 从 only_train_once.transform 模块导入 tensor_transformation 函数和 TensorTransform 枚举
import numpy as np                                                                  						# 导入 numpy 库，用于数值计算

# 定义一个抽象基类 BasicNodeGroup，表示一个基本的节点组
class BasicNodeGroup(ABC):
    # 初始化方法
    def __init__(self, is_prunable=True):
        self.nodes = dict()                         # 存储节点组中的节点，以节点ID为键，节点对象为值
        self.output_nodes = dict()                  # 存储节点组中的输出节点
        self.is_prunable = is_prunable              # 标记该节点组是否可剪枝
        self.pruning_redundant_idxes = list()       # 存储剪枝过程中冗余的索引
        self.pruning_important_idxes = list()       # 存储剪枝过程中重要的索引
        self.is_auxiliary = False                   # 标记该节点组是否是辅助节点组
        self.extra_param_group_attrs = dict()       # 存储额外的参数组属性

    # 定义对象的字符串表示形式
    def __repr__(self):
        return f"Id: {self.id}, is_prunable: {self.is_prunable}, nodes: {self.nodes}"
    
    # 返回节点组中节点的数量
    def num_nodes(self):
        return len(self.nodes)
    
    # 定义一个属性，返回节点组的ID，由组内所有节点的ID用下划线连接而成
    @property
    def id(self):
        return "_".join([node.id for node in self.nodes.values()])
        
    #向节点组中添加一个节点
    def add_node(self, node):
        # 如果节点ID尚未在节点组中，则添加
        if node.id not in self.nodes:
            self.nodes[node.id] = node

    # 向节点组中添加多个节点
    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)

    # 检查节点组是否包含给定节点列表中的任何一个节点
    def contain_some_nodes(self, nodes):
        for node in nodes:
            if self.contain_node(node):
                return True
        return False
        
    # 检查节点组是否包含指定的节点
    def contain_node(self, node):
        # 如果节点ID在节点组中，则返回True
        return True if node.id in self.nodes else False
    
    def contain_op(self, op_name):
        for node in self.nodes.values():
            if not node.op:
                continue
            if op_name == type(node.op).__name__:
                return True
        return False
    
	# 从节点组中移除一个节点
    def remove_node(self, node):
        # 如果节点ID在节点组中，则删除该节点
        if node.id in self.nodes:
            del self.nodes[node.id]
    
    # 抽象方法，获取参数组，需要在子类中实现
    @abstractclassmethod
    def get_param_groups(self):
        raise NotImplementedError

    # 定义一个属性，返回节点组中所有参数的名称
    @property
    def param_names(self):
        return self.get_param_names()
    
    # 获取节点组中所有参数的名称
    def get_param_names(self):    
        param_names = list()
        # 遍历节点组中的每个节点
        for node in self:
            # 如果节点没有参数名称，则跳过
            if len(node.param_names) == 0:
                continue
            param_names.extend(node.param_names)        # 将节点的参数名称添加到列表中
        return param_names

    # 定义迭代器的初始化方法
    def __iter__(self):
        self._iter_idx = 0                              # 初始化迭代索引
        self._node_ids = list(self.nodes.keys())        # 获取节点组中所有节点的ID列表
        return self
    
    # 定义迭代器的下一个元素获取方法
    def __next__(self):
        # 如果迭代索引小于节点数量
        if self._iter_idx < self.num_nodes():
            node = self.nodes[self._node_ids[self._iter_idx]]       # 获取当前索引对应的节点
            self._iter_idx += 1                                     # 迭代索引加1
            return node
        # 如果迭代完成，则抛出 StopIteration 异常
        raise StopIteration
    
    # 设置节点组的输出节点
    def set_output_nodes(self, graph):
        # 遍历节点组中的所有节点
        for node in self.nodes.values():
            is_out_node = True
            # 遍历节点的每个出向节点
            for node_out in graph.outgoing(node):
                # 如果出向节点也在当前节点组中，则当前节点不是输出节点
                if node_out.id in self.nodes:
                    is_out_node = False
            # 如果当前节点是输出节点，则将其添加到 output_nodes 字典中
            if is_out_node:
                self.output_nodes[node.id] = node 

    # 获取节点组中所有节点的ID，skip_output_node: 是否跳过输出节点
    def get_node_ids(self, skip_output_node=False):
        # 如果不跳过输出节点，则返回所有节点的ID集合，否则，返回所有节点ID集合与输出节点ID集合的差集
        return set(self.nodes.keys()) if not skip_output_node \
            else set(self.nodes.keys()).difference(self.output_nodes.keys())

    # 合并另一个节点组到当前节点组
    def merge(self, node_group):
        # 将另一个节点组中的所有节点添加到当前节点组
        self.add_nodes(node_group.nodes.values())

        
class NodeGroup(BasicNodeGroup):
    # NodeGroup 类继承自 BasicNodeGroup，代表一个更具体的节点组实现。
    # 它负责管理节点组内的节点，并提供计算组数量、获取模块、获取参数组、设置剪枝索引等功能。
    def __init__(self, is_prunable=True):
        # 调用父类 BasicNodeGroup 的初始化方法。
        # is_prunable: 布尔值，指示此节点组是否可以被剪枝。
        super().__init__(is_prunable)       # 调用父类的初始化方法

    @property
    def num_groups(self):
        # 计算并返回节点组中的组数量。组数量由节点组内所有节点操作（op）的 num_groups 属性的最大值决定。
        # 如果节点没有参数或操作，则在计算中被忽略。
        num_groups = 1      # 初始化组数量为1，至少为1组
        # 遍历节点组中的每个节点 (通过 __iter__ 方法)
        for node in self: 
            # 如果节点没有参数名称 (例如，某些操作节点可能没有直接关联的可训练参数)，则跳过
            if len(node.param_names) == 0: 
                continue
            # 如果节点没有定义操作 (op)，则跳过
            if not node.op: 
                continue
            # 更新 num_groups 为当前 num_groups 和节点操作的组数量中的较大者。这确保了 num_groups 反映了组内所有操作所需的最大分组数。
            num_groups = max(num_groups, node.op.num_groups) 
        # 返回计算得到的最大组数量
        return num_groups 

    def get_modules(self):
        # 获取并返回节点组中所有唯一的 PyTorch 模块 (nn.Module)。
        # 遍历节点组中的所有节点，如果节点的操作 (op) 具有 'module' 属性，则将该模块添加到集合中，以确保唯一性。
        modules = set()         # 初始化一个空集合用于存储模块，集合自动处理重复项
        # 遍历节点组中的每个节点
        for node in self: 
            # 如果节点没有操作，则跳过
            if not node.op: 
                continue
            # 检查节点的操作对象是否具有 'module' 属性
            if hasattr(node.op, 'module'): 
                modules.add(node.op.module)         # 将模块添加到集合中
        return modules                              # 返回包含所有唯一模块的集合
    
    def get_param_groups(self):
        ng_param_group = dict()
        ng_param_group['id'] = self.id
        ng_param_group['num_groups'] = self.num_groups
        ng_param_group['is_prunable'] = self.is_prunable
        ng_param_group['is_auxiliary'] = self.is_auxiliary
        ng_param_group['p_names'] = list()
        ng_param_group['params'] = list()
        ng_param_group['op_names'] = list()
        ng_param_group['p_transform'] = list()
        ng_param_group['auxiliary_ngs'] = list()
        ng_param_group['node_ids'] = list()

        basic_attrs = ['op_names', 'p_names', 'params', 'p_transform']
        for node in self:
            if len(node.param_names) == 0 or not node.op:
                continue
            node_param_groups = node.op.get_param_groups(param_names=node.param_names)
            ng_param_group['op_names'].extend([node_param_groups['op']] * len(node_param_groups['params']))
            ng_param_group['p_names'].extend(node_param_groups['p_names'])
            ng_param_group['params'].extend(node_param_groups['params'])
            ng_param_group['p_transform'].extend(node_param_groups['p_transform'])
            ng_param_group['node_ids'].extend([node.id for _ in node_param_groups['p_names']])
            for attr in node_param_groups:
                if attr not in basic_attrs:
                    ng_param_group[attr] = node_param_groups[attr]
        assert len(ng_param_group['params']) == len(ng_param_group['p_names'])
        for attr in self.extra_param_group_attrs:
            if attr not in ng_param_group:
                ng_param_group[attr] = self.extra_param_group_attrs[attr]
        return ng_param_group


    def set_pruning_redundant_idxes(self):
        param_groups = self.get_param_groups()
        if len(param_groups['params']) == 0 and not self.is_auxiliary:
            self.pruning_important_idxes, self.pruning_redundant_idxes = list(), list()
            return 
        elif len(param_groups['params']) > 0 and not self.is_auxiliary:
            norm_group = None
            for (param, p_transform) in zip(param_groups['params'], param_groups['p_transform']):
                if p_transform == TensorTransform.NO_PRUNE:
                    continue
                
                param_transform = None        
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(param, p_transform, param_groups['num_groups'], param_groups['num_heads'])
                elif isinstance(p_transform, list):
                    param_transform = param.data.clone()
                    for (p_transform_type, p_transform_config) in p_transform:
                        if p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD or p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
                            param_transform = tensor_transformation(param_transform, p_transform_type, p_transform_config['num_heads'])
                        elif p_transform_type == TensorTransform.MULTIHEAD_HEADDIM:
                            param_transform = tensor_transformation(param_transform, p_transform_type, p_transform_config['head_dim'], p_transform_config['num_heads'])
                        else:
                            param_transform = tensor_transformation(param_transform, p_transform_type, num_groups=p_transform_config['num_groups'])
                else:
                    param_transform = tensor_transformation(param, p_transform, param_groups['num_groups'])

                if norm_group == None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2
            if norm_group is None:
                self.pruning_important_idxes, self.pruning_redundant_idxes = list(), list()
                return
            norm_group = torch.sqrt(norm_group)
            norm_group = norm_group.cpu()

            self.pruning_important_idxes = np.arange(self.num_groups)[norm_group != 0]
            self.pruning_redundant_idxes = np.arange(self.num_groups)[norm_group == 0]

            # TODO: index list transformation
            if hasattr(self, 'overwrite_p_transform'):
                # if self.overwrite_p_transform == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD and 'head_dim' in param_groups:
                #     head_dim = param_groups['head_dim']
                #     # head_dim = p_transform_config['head_dim'] 
                #     refined_pruning_important_idxes = index_transformation(self.pruning_important_idxes, p_transform_type, head_dim=head_dim)
                #     refined_pruning_redundant_idxes = index_transformation(self.pruning_redundant_idxes, p_transform_type, head_dim=head_dim)
                #     self.pruning_important_idxes = np.array(refined_pruning_important_idxes)
                #     self.pruning_redundant_idxes = np.array(refined_pruning_redundant_idxes)
                if self.overwrite_p_transform == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD and 'head_dim' in param_groups:
                    head_dim = param_groups['head_dim']
                    refined_pruning_important_idxes = list()
                    refined_pruning_important_idxes = index_transformation(self.pruning_important_idxes, TensorTransform.MULTIHEAD_NUMHEAD, head_dim=head_dim)
                    refined_pruning_redundant_idxes = index_transformation(self.pruning_redundant_idxes, TensorTransform.MULTIHEAD_NUMHEAD, head_dim=head_dim)
                    # for i in self.pruning_important_idxes:
                    #     refined_pruning_important_idxes.extend([h + i * head_dim for h in range(head_dim)])
                    self.pruning_important_idxes = np.array(refined_pruning_important_idxes)
                    # refined_pruning_redundant_idxes = list()
                    # for i in self.pruning_redundant_idxes:
                    #     refined_pruning_redundant_idxes.extend([h + i * head_dim for h in range(head_dim)])
                    self.pruning_redundant_idxes = np.array(refined_pruning_redundant_idxes)
                elif isinstance(self.overwrite_p_transform, list):
                    refined_pruning_important_idxes = [i for i in self.pruning_important_idxes]
                    refined_pruning_redundant_idxes = [i for i in self.pruning_redundant_idxes]
                    for (p_transform_type, p_transform_config) in reversed(self.overwrite_p_transform):
                        if p_transform_type == TensorTransform.MULTIHEAD_HEADDIM:
                            head_dim = p_transform_config['head_dim']
                            num_heads = p_transform_config['num_heads']
                            refined_pruning_important_idxes = index_transformation(refined_pruning_important_idxes, p_transform_type, num_heads=num_heads, head_dim=head_dim)
                            refined_pruning_redundant_idxes = index_transformation(refined_pruning_redundant_idxes, p_transform_type, num_heads=num_heads, head_dim=head_dim)
                        elif p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD or p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
                            head_dim = p_transform_config['head_dim'] 
                            refined_pruning_important_idxes = index_transformation(refined_pruning_important_idxes, p_transform_type, head_dim=head_dim)
                            refined_pruning_redundant_idxes = index_transformation(refined_pruning_redundant_idxes, p_transform_type, head_dim=head_dim)
                        self.pruning_important_idxes = np.array(refined_pruning_important_idxes)
                        self.pruning_redundant_idxes = np.array(refined_pruning_redundant_idxes)
        elif self.is_auxiliary:
            pruning_redundant_idxes = list()
            offset = 0
            for dependent_node_group in self.dependent_node_groups:
                if len(dependent_node_group.pruning_redundant_idxes) == 0:
                    offset += dependent_node_group.num_groups
                    continue
                pruning_redundant_idxes.append(dependent_node_group.pruning_redundant_idxes + offset)
                offset += (dependent_node_group.pruning_important_idxes.size + dependent_node_group.pruning_redundant_idxes.size)
            if len(pruning_redundant_idxes) > 0:
                self.pruning_redundant_idxes = np.concatenate(pruning_redundant_idxes)
            else:
                self.pruning_redundant_idxes = list()
            self.pruning_important_idxes = list()

        for (param, p_transform, node_id) in zip(param_groups['params'], param_groups['p_transform'], param_groups['node_ids']):
            if p_transform == TensorTransform.NO_PRUNE:
                continue
            node = self.nodes[node_id]
            if isinstance(p_transform, list):
                refined_pruning_important_idxes = [i for i in self.pruning_important_idxes]
                refined_pruning_redundant_idxes = [i for i in self.pruning_redundant_idxes]
                for (p_transform_type, p_transform_config) in reversed(p_transform):
                    if p_transform_type == TensorTransform.MULTIHEAD_HEADDIM:
                        head_dim = p_transform_config['head_dim']
                        num_heads = p_transform_config['num_heads']
                        refined_pruning_important_idxes = index_transformation(refined_pruning_important_idxes, p_transform_type, num_heads=num_heads, head_dim=head_dim)
                        refined_pruning_redundant_idxes = index_transformation(refined_pruning_redundant_idxes, p_transform_type, num_heads=num_heads, head_dim=head_dim)
                    elif p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD or p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
                        head_dim = p_transform_config['head_dim'] 
                        refined_pruning_important_idxes = index_transformation(refined_pruning_important_idxes, p_transform_type, head_dim=head_dim)
                        refined_pruning_redundant_idxes = index_transformation(refined_pruning_redundant_idxes, p_transform_type, head_dim=head_dim)
                    node.pruning_important_idxes = np.array(refined_pruning_important_idxes)
                    node.pruning_redundant_idxes = np.array(refined_pruning_redundant_idxes)
            else:
                node.pruning_important_idxes = self.pruning_important_idxes
                node.pruning_redundant_idxes = self.pruning_redundant_idxes

    def prune_out_dim(self, global_skip_modules=set()):
        local_skip_modules=set()    
        for node in self.nodes.values():
            if not node.op:
                continue
            if hasattr(node.op, 'prune_out_dim'):
                if node.op.module not in local_skip_modules and node.op.module not in global_skip_modules:
                    node.op.prune_out_dim(pruned_idxes=self.pruning_redundant_idxes, param_names=node.param_names)
                    local_skip_modules.add(node.op.module)
                elif node.op.module is None and type(node.op).__name__ == 'ParamOTO':
                    # ParamOperator does not have module
                    node.op.prune_out_dim(pruned_idxes=self.pruning_redundant_idxes, param_names=node.param_names)
                    local_skip_modules.add(node.op.param)
                elif self.contain_lora():
                    node.op.prune_out_dim(pruned_idxes=self.pruning_redundant_idxes, param_names=node.param_names)
                    
    def contain_lora(self):
        for node in self:
            if len(node.param_names) == 0 or not node.op:
                continue
            for param_name in node.param_names:
                if 'lora_B' in param_name or 'lora_embedding_B' in param_name:
                    self.scaling = node.op.lora_scaling
                    return True
        return False
    
    def contain_stem_op(self):
        is_stem = False
        for node in self:
            if not node.op:
                continue
            if node.op.is_stem:
                is_stem = True
        return is_stem

    def contain_concat(self, axis=1):
        for node in self:
            if node.is_concat(axis=axis):
                return True
        return False

    def get_concat_nodes(self, axis=1):
        concat_nodes = list()
        for node in self:
            if node.is_concat(axis=axis):
                concat_nodes.append(node)
        return concat_nodes
        
    def set_auxiliary(self):
        if self.contain_concat(axis=1):
            self.is_auxiliary = True
            return True
        else:
            self.is_auxiliary = False
            return False


class NodeGroupComposedOp(BasicNodeGroup):
    """
    NodeGroupComposedOp refers to the node group for a composed operator
    """
    def __init__(self, is_prunable=True, op=None):
        super().__init__(is_prunable)
        self.op = op

    def get_modules(self):
        modules = set()
        if not self.op:
            return modules
        elif hasattr(self.op, 'module'):
            modules.add(self.op.module)
        return modules
                        
    def set_node_equivalence(self):
        self.node_cluster_by_leaf_module = dict()
        self.node_id_to_leaf_module_id = dict()
        for node in self:
            if len(node.param_names) == 0:
                continue
            for leaf_module in self.op.leaf_modules.values():
                if set(node.param_names).issubset(set(leaf_module.param_names)):
                    if leaf_module.id not in self.node_cluster_by_leaf_module:
                        self.node_cluster_by_leaf_module[leaf_module.id] = list()
                    self.node_cluster_by_leaf_module[leaf_module.id].append(node)
                    self.node_id_to_leaf_module_id[node.id] = leaf_module.id

    def set_auxiliary(self):
        # TODO: to implemented.
        self.is_auxiliary = False

    def set_output_nodes(self, graph):
        for node in self.nodes.values():
            is_out_node = True
            for node_out in graph.outgoing(node):
                if node_out.id in self.nodes:
                    is_out_node = False 
            if is_out_node:
                self.output_nodes[node.id] = node 
        
        new_node_outs = set()
        for node_out in self.output_nodes.values():
            if node_out.id not in self.node_id_to_leaf_module_id:
                continue
            leaf_module_id = self.node_id_to_leaf_module_id[node_out.id]
            node_cluster = self.node_cluster_by_leaf_module[leaf_module_id]
            for node in node_cluster:
                if node.id not in self.output_nodes:
                    new_node_outs.add(node)

        # needs to include the incoming stems for the direct output_node
        visited = dict.fromkeys(self.nodes, False)
        def dfs_helper(node, graph, path):
            if node.is_stem():
                for node_new in path:
                    if node_new.id not in self.output_nodes:
                        new_node_outs.add(node_new)
                return 
            for node_in in graph.incoming(node):
                if node_in.id in self.nodes and not visited[node_in.id]:
                    visited[node_in.id] = True
                    path.append(node_in)
                    dfs_helper(node_in, graph, path)

        for out_node in self.output_nodes.values():
            dfs_helper(out_node, graph, [])

        for node in new_node_outs:
            self.output_nodes[node.id] = node 

        self.out_param_names = list()
        for out_node in self.output_nodes.values():
            self.out_param_names.extend(out_node.param_names)
        # Set op out_param_names for composed op
        self.op.out_param_names = self.out_param_names

    def get_param_groups(self):
        ng_param_group = dict()
        ng_param_group['id'] = self.id
        ng_param_group['num_groups'] = self.num_groups
        ng_param_group['is_prunable'] = self.is_prunable
        ng_param_group['is_auxiliary'] = self.is_auxiliary
        ng_param_group['p_names'] = list()
        ng_param_group['params'] = list()
        ng_param_group['op_names'] = list()
        ng_param_group['p_transform'] = list()
        ng_param_group['auxiliary_ngs'] = list()
        
        # Skip the output node params, which should depend on other node groups
        op_param_group = self.op.get_param_groups(skip_output_node=True)
        basic_attrs = ['op_names', 'p_names', 'params', 'p_transform']
        ng_param_group['op_names'].extend([op_param_group['op']] * len(op_param_group['params']))
        ng_param_group['p_names'].extend(op_param_group['p_names'])
        ng_param_group['params'].extend(op_param_group['params'])
        ng_param_group['p_transform'].extend(op_param_group['p_transform'])
        
        # 遍历操作对象的参数组属性
        for attr in op_param_group:  
            # 如果属性不在基本属性列表中
            if attr not in basic_attrs:  
                ng_param_group[attr] = op_param_group[attr]     # 将属性添加到参数组字典中

        # 遍历额外的参数组属性
        for attr in self.extra_param_group_attrs:  
            # 如果属性不在参数组字典中
            if attr not in ng_param_group:  
                ng_param_group[attr] = self.extra_param_group_attrs[attr]   # 将属性添加到参数组字典中
        return ng_param_group  # 返回参数组字典

    @property
    def num_groups(self):
        """
        获取组数量。
        返回:
        - (int): 组数量。
        """
        return self.op.num_groups  # 返回操作对象的组数量

    def set_pruning_redundant_idxes(self):
        """
        设置剪枝的冗余索引。
        """
        param_groups = self.get_param_groups()      # 获取参数组信息
        # 如果没有参数且不是辅助节点组
        if len(param_groups['params']) == 0 and not self.is_auxiliary:  
            self.pruning_important_idxes, self.pruning_redundant_idxes = list(), list()     # 初始化重要和冗余索引为空列表
            return
        # 如果有参数且不是辅助节点组
        elif len(param_groups['params']) > 0 and not self.is_auxiliary:  
            norm_group = None  # 初始化范数组
            for (p_name, param, p_transform) in zip(param_groups['p_names'], param_groups['params'], param_groups['p_transform']):  # 遍历参数
                if p_transform == TensorTransform.NO_PRUNE:
                    continue
                if 'lora_A' in p_name or 'lora_embedding_A' in p_name:  # 跳过特定参数
                    continue
                param_transform = None  # 初始化参数转换
                # 如果是多头注意力机制的头维度转换
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:  
                    param_transform = tensor_transformation(param, p_transform, param_groups['num_groups'], param_groups['num_heads'])
                elif isinstance(p_transform, list):
                    param_transform = param.data.clone() 
                    for (p_transform_type, p_transform_config) in p_transform:
                        if p_transform_type == TensorTransform.MULTIHEAD_HEADDIM:
                            head_dim = p_transform_config['head_dim']
                            num_heads = p_transform_config['num_heads']
                            param_transform = tensor_transformation(param_transform, p_transform_type, num_heads=num_heads, num_groups=head_dim)
                        elif p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD or p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
                            num_heads = p_transform_config['num_heads']
                            param_transform = tensor_transformation(param_transform, p_transform_type, num_groups=num_heads)
                else:
                    param_transform = tensor_transformation(param, p_transform, param_groups['num_groups'])
                # 如果范数组为空
                if norm_group == None: 
                    norm_group = torch.norm(param_transform, dim=1) ** 2    # 计算范数平方
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2   # 累加范数平方
            norm_group = torch.sqrt(norm_group)                             # 计算范数平方根
            norm_group = norm_group.cpu()                                   # 将范数组移动到CPU
            if self.num_groups == 1:
                self.pruning_important_idxes = np.arange(1)
                self.pruning_redundant_idxes = []
            else:
                self.pruning_important_idxes = np.arange(self.num_groups)[norm_group != 0]
                self.pruning_redundant_idxes = np.arange(self.num_groups)[norm_group == 0]

    def prune_out_dim(self, **kwargs):
        """
        根据冗余索引剪枝输出维度。
        """
        # 如果操作对象具有 prune_out_dim 方法
        if hasattr(self.op, 'prune_out_dim'):  
            self.op.prune_out_dim(pruned_idxes=self.pruning_redundant_idxes, skip_output_node=True)  # 调用剪枝方法
            # 遍历节点组中的每个节点
            for node in self:  
                # 如果节点是输出节点
                if node.id in self.output_nodes:  
                    continue
                node.pruned_status['out_dim'] = True  # 设置节点的剪枝状态

    def contain_lora(self):
        """
        检查节点组是否包含 LoRA 结构。
        返回:
        - (bool): 如果包含 LoRA 结构，则返回 True；否则返回 False。
        """
        # 遍历节点组中的每个节点
        for node in self:  
            # 如果节点没有参数或操作，则跳过
            if len(node.param_names) == 0 or not node.op:  
                continue
            # 遍历节点的参数名称
            for param_name in node.param_names:  
                # 如果参数名称包含 'lora_B'
                if 'lora_B' in param_name:  
                    self.scaling = node.op.lora_scaling  # 设置 LoRA 的缩放因子
                    return True  # 返回 True
        return False  # 如果没有找到 LoRA 结构，则返回 False

    def contain_stem_op(self):
        """
        检查节点组是否包含 stem 操作。
        返回:
        - (bool): 如果包含 stem 操作，则返回 True；否则返回 False。
        """
        return self.op.is_stem  # 返回操作对象的 is_stem 属性

    def set_auxilary(self):
        """
        设置辅助节点组。
        当前未实现。
        """
        pass
    