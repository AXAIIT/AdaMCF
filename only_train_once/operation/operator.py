import torch
import torch.nn as nn
from only_train_once.transform import TensorTransform, index_transformation # 导入张量变换相关的类
from abc import ABC, abstractclassmethod # 导入抽象基类和抽象方法装饰器

# 定义基础算子类，作为所有具体算子的父类
class BasicOperator(ABC):
    # 初始化方法
    def __init__(self, id=None, _type=None, cfg_params=dict()):
        """
        初始化基础算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
        """
        self.id = id                    # 算子唯一标识符
        self._type = _type              # 算子类型
        self.cfg_params = cfg_params    # 算子配置参数
        self.is_stem = False            # Stem 算子可以转换输入张量的主维度（通常是通道维度），标记是否为 Stem 算子，默认为 False
        # 记录输入和输出维度是否已被剪枝的状态
        self.pruned_status = {
            'out_dim': False,           # 输出维度剪枝状态
            'in_dim': False             # 输入维度剪枝状态
        }

    # 抽象方法，子类必须实现此方法以返回参数组
    @abstractclassmethod
    def get_param_groups(self):
        """
        获取算子的参数组。这是一个抽象方法，需要在子类中实现。
        通常用于优化器配置或特定参数处理。
        """
        raise NotImplementedError       # 如果子类未实现，则抛出 NotImplementedError

    def prune_param_and_grad(self, param, preserved_idxes, dim=0):
        pruned_param = torch.nn.Parameter(torch.index_select(param, dim, torch.LongTensor(preserved_idxes).to(param.device)))
        if param.grad is not None:
            pruned_param.grad = torch.index_select(param.grad, dim, torch.LongTensor(preserved_idxes).to(param.device))
        return pruned_param.to(param.device)


class Operator(BasicOperator):
    """
    通用的算子类，继承自 BasicOperator。
    封装了 PyTorch 的 nn.Module，并提供了额外的管理功能，
    如叶子模块识别、参数分组、FLOPs 计算等。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化 Operator。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (torch.nn.Module, optional): 关联的 PyTorch 模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params)         # 调用父类的初始化方法
        self.module = module                            # 关联的 PyTorch 模块
        self.leaf_modules = dict()                      # 存储叶子模块（基础或复合算子）的字典

        self.set_leaf_modules()                         # 递归查找并设置叶子模块
        self.set_param_names()                          # 设置算子及其子模块的参数名称列表
        self.name_to_param = dict()                     # 映射：参数名称 -> 参数对象
        # 填充 name_to_param 字典，使用完整的参数名称（包括算子ID）
        for name, param in self.named_parameters():
            self.name_to_param[self.id+'.'+name] = param
        self.num_groups = 1                             # 参数分组的数量，用于某些类型的剪枝或变换
        self.is_basic = True                            # 标记是否为基础模块（相对于复合模块）
        self.p_transform = TensorTransform.BASIC        # 参数的默认变换类型

    def __eq__(self, name):
        """
        重载等于运算符，用于通过名称比较算子。
        """
        return self.name == name 

    def __repr__(self) -> str:
        """
        返回算子的详细信息字符串表示。
        """
        return self._full_info()

    def _get_module_type(self, module):
        """
        获取给定 PyTorch 模块的类型名称字符串。
        """
        return type(module).__name__

    def _full_info(self):
        """
        生成包含算子ID、类型、叶子模块键和参数名称的完整信息字符串。
        """
        return "Id: {id}, Type: {type}, Leaf Modules: {leaf_module_keys}, Param Names: {param_names}".format(
            id=self.id, type=self._type, leaf_module_keys=" ".join(list(self.leaf_modules.keys())), param_names=" ".join(self.param_names)
        )

    def set_leaf_modules(self):
        """
        递归地遍历关联的PyTorch模块 (self.module)，识别并存储叶子模块。
        叶子模块可以是预定义的 BASIC_MODULES 或 COMPOSED_MODULES。
        """
        if not self.module: # 如果没有关联模块，则直接返回
            return
        # 定义深度优先搜索 (DFS) 辅助函数
        def dfs_helper(module, module_name, composed_op):
            module_type = self._get_module_type(module) # 获取当前模块类型
            # 检查是否为预定义的复合模块
            if module_type in COMPOSED_MODULES:
                # 创建对应的复合算子实例
                composed_op = COMPOSED_MODULES[module_type](
                    id = module_name,
                    _type = module_type,
                    module = module)
                # 将复合算子添加到叶子模块字典中
                self.leaf_modules[composed_op.id] = composed_op
                return # 复合模块是叶子节点，停止向下递归

            # 检查当前模块是否没有子模块（即PyTorch模块层面的叶子节点）
            if next(module.named_children(), None) is None:
                # 检查是否为预定义的基础模块
                if module_type in BASIC_MODULES:
                    # 创建对应的基础算子实例并添加到叶子模块字典
                    self.leaf_modules[module_name] = BASIC_MODULES[module_type](
                        id = module_name,
                        _type = module_type,
                        module = module)
                return # 到达叶子节点，停止递归

            # 如果当前模块有子模块，则递归遍历子模块
            for name, module_child in module.named_children():
                # 递归调用 dfs_helper，构建子模块的完整名称
                dfs_helper(module_child, module_name + '.' + name if module_name != '' else name, composed_op)

        # 如果顶级模块本身没有子模块，则将自身视为叶子模块
        if next(self.module.named_children(), None) is None:
            self.leaf_modules[self.id] = self
        else: # 否则，从顶级模块的子模块开始 DFS
            for name, module_child in self.module.named_children():
                # 构建子模块的完整名称并开始 DFS
                dfs_helper(module_child, self.id + '.' + name if self.id != '' else name, None)

    def set_param_names(self):
        """
        收集算子及其关联模块的所有参数名称，并存储在self.param_names列表中。
        参数名称包含算子的ID作为前缀。
        """
        self.param_names = list() # 初始化参数名称列表
        # 如果没有关联模块，则直接返回
        if not self.module: 
            return
        # 遍历模块的所有命名参数
        for name, _ in self.module.named_parameters():
            # 将带有算子 ID 前缀的参数名称添加到列表中
            self.param_names.append(self.id + '.' + name)

    def named_parameters(self):
        """
        返回关联模块的命名参数生成器。
        如果无关联模块，则返回空列表。
        """
        return self.module.named_parameters() if self.module else []

    def get_param_groups(self, param_names=list(), **kwargs):
        """
        根据提供的参数名称列表，为该算子创建参数组。
        参数组通常用于优化器，允许对不同类型的参数应用不同的设置。
        Args:
            param_names (list, optional): 需要包含在参数组中的参数名称列表。如果为空，则默认不包含任何参数（具体行为可能因子类而异）。
            **kwargs: 其他可选参数。
        Returns:
            dict: 包含参数组信息的字典，结构如下：
                  {
                      'op': str,          # 算子类型
                      'p_names': list,    # 参数名称列表
                      'params': list,     # 参数对象列表
                      'p_transform': list # 参数对应的变换类型列表
                  }
        """
        param_groups = dict()                   # 初始化参数组字典
        param_groups['op'] = self._type         # 设置算子类型
        param_groups['p_names'] = list()        # 初始化参数名称列表
        param_groups['params'] = list()         # 初始化参数对象列表
        param_groups['p_transform'] = list()    # 初始化参数变换类型列表
        # 遍历指定的参数名称
        for p_name in param_names:
            param = self.name_to_param[p_name]  # 从 name_to_param 字典中获取参数对象
            if not param.requires_grad:         # 如果参数不需要梯度，则跳过
                continue
            # 将参数名称、参数对象和变换类型添加到相应的列表中
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(param)
            param_groups['p_transform'].append(self.p_transform)
        return param_groups                     # 返回参数组字典

    def set_num_groups(self):
        """
        设置算子的num_groups属性。
        通常基于参数的形状（例如，卷积核的输出通道数）来确定分组数量。
        默认实现是取所有参数第 0 维大小的最大值。
        子类可以覆盖此方法以实现特定的分组逻辑。
        """
        self.num_groups = 1                     # 初始化为 1
        # 遍历所有参数
        for param_name in self.name_to_param:
            param = self.name_to_param[param_name]
            # 更新 num_groups 为当前最大值和参数第 0 维大小中的较大者
            # 注意：这里假设第 0 维代表分组维度，这可能不适用于所有层类型
            self.num_groups = max(self.num_groups, param.shape[0])

    def compute_flops(self, input_shape):
        """
        计算算子的浮点运算次数 (FLOPs)。
        默认实现返回 0。子类应覆盖此方法以提供准确的 FLOPs 计算。
        Args:
            input_shape (tuple): 输入张量的形状。
        Returns:
            int: 算子的 FLOPs。
        """
        return 0 # 默认返回 0

    @property
    def num_params(self):
        """
        计算并返回算子包含的总参数数量（可训练和不可训练）。
        """
        num_params = 0 # 初始化参数计数器
        # 遍历所有参数
        for param_name in self.name_to_param:
            param = self.name_to_param[param_name]
            num_params += param.numel() # 累加参数的元素数量
        return num_params               # 返回总参数数量

class ParamOTO(Operator):
    '''
    用于管理尚未封装到 nn.Module 中的 PyTorch 张量参数的算子。
    该类负责处理独立的张量参数（例如在某些网络设计中单独创建的权重参数），
    使其能够像普通的模块参数一样被剪枝、优化等操作处理。
    '''
    def __init__(self, id=None, _type=None, cfg_params=dict(), param_name=None, param=None):
        '''
        初始化 ParamOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            param_name (str, optional): 参数的名称。默认为 None。
            param (torch.Tensor, optional): 实际的张量参数。默认为 None。
        '''
        super().__init__(id, _type, cfg_params)     # 调用父类 Operator 的初始化方法，对应基础算子的通用初始化逻辑
        self.is_stem = False                        # 标记为非主干(Stem)算子（不转换输入主维度）
        self.param_name = param_name                # 记录参数名称，用于参数管理和引用
        self.param = param                          # 存储实际的张量参数对象

    def get_param_groups(self, **kwargs):
        '''
        获取该算子的参数分组信息，用于优化器配置。
        Returns:
        dict: 包含参数分组信息的字典，结构如下：
            {
                'op': str,          # 算子类型
                'p_names': list,    # 参数名称列表
                'params': list,     # 参数对象列表
                'p_transform': list # 参数变换类型列表
            }
        '''
        # 返回包含该算子管理的参数分组信息
        param_groups = dict()                               # 初始化参数分组字典
        param_groups['op'] = self._type                     # 算子类型
        param_groups['p_names'] = [self.param_name]         # 参数名称列表（只包含一个参数）
        param_groups['params'] = [self.param]               # 实际张量参数对象
        param_groups['p_transform'] = [self.p_transform]    # 应用于参数的变换类型
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝参数的输出维度（param的第0维）。
        通过指定要移除的索引，保留其余部分，实现对参数的剪枝操作。
        Args:
            pruned_idxes (list, optional): 需要被剪去的索引列表。默认为空列表。
            **kwargs: 额外的关键字参数，传递给底层剪枝函数。  
        注意:
            - 此方法假设输出维度对应参数张量的第 0 维
            - 剪枝会同时处理参数及其梯度（如果存在）
        """
        preserved_idxes = list(set(range(self.param.shape[0])) - set(pruned_idxes))     # 计算保留的索引（所有索引减去要剪去的索引）
        preserved_idxes.sort()                                                          # 确保索引有序，保持维度的原始相对顺序
        
        # 调用基类方法 prune_param_and_grad，根据保留索引剪枝参数及其梯度，dim=0 表示在第 0 维度上进行剪枝
        self.param = self.prune_param_and_grad(self.param, preserved_idxes, 0)
        
class Conv2dOTO(Operator):
    """
    卷积2D算子类，封装并扩展了PyTorch的nn.Conv2d模块。
    提供了参数管理、通道维度剪枝和计算量估计等功能。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化Conv2dOTO算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为None。
            _type (str, optional): 算子的类型名称。默认为None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.Conv2d, optional): 关联的PyTorch卷积模块。默认为None。
        """
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = True         # 标记为Stem算子，表示可以转换输入张量的主维度
        self.set_num_groups()       # 设置分组数，通常为输出通道数
    
    def get_param_groups(self, param_names=list()):
        """
        获取卷积算子的参数分组信息，用于优化器配置。
        Args:
            param_names (list, optional): 要包含在参数组中的参数名称列表。默认为空列表。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()
        param_groups['op'] = 'conv2d'           # 设置操作类型为卷积2D
        param_groups['p_names'] = list()        # 初始化参数名称列表
        param_groups['params'] = list()         # 初始化参数对象列表
        param_groups['p_transform'] = list()    # 初始化参数变换类型列表
        # 遍历指定的参数名称，收集对应的参数和变换类型
        for p_name in param_names:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝卷积算子的输出通道维度。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输出通道索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        注意:
            - 对于组卷积(groups=out_channels)的特殊情况有专门处理
            - 同时更新模块的out_channels属性和权重(以及偏置)参数
        """
        # 处理特殊情况：组卷积且组数等于输出通道数（如深度可分离卷积）
        if self.module.groups == self.module.out_channels:
            # 更新组数为剪枝后的输出通道数
            self.module.groups = self.module.out_channels - len(pruned_idxes)
        
        preserved_idxes = list(set(range(self.module.out_channels)) - set(pruned_idxes))        # 计算需要保留的输出通道索引
        preserved_idxes.sort()  # 对索引排序，保持相对顺序
        self.module.out_channels = self.module.out_channels - len(pruned_idxes)                 # 更新模块的输出通道数
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)  # 剪枝权重参数，在输出维度(dim=0)上进行剪枝
        # 如果存在偏置，也对其进行剪枝
        if self.module.bias is not None:
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)

    def prune_in_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝卷积算子的输入通道维度。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输入通道索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        注意:
            - 对于组卷积(groups=out_channels)的特殊情况，直接返回不进行剪枝
            - 如果剪枝后的输入维度不合法（大于等于当前权重形状），直接返回
        """
        preserved_idxes = list(set(range(self.module.in_channels)) - set(pruned_idxes))     # 计算需要保留的输入通道索引
        preserved_idxes.sort()                                                              # 对索引排序，保持相对顺序
        # 特殊情况处理：组卷积且组数等于输出通道数，此时不进行输入维度剪枝
        # 见zig.py中的详细解释
        if self.module.groups == self.module.out_channels and self.module.groups > 1:
            return 
        # 如果剪枝后的输入维度不合理，直接返回
        if self.module.weight.shape[1] <= len(preserved_idxes):
            return
        # 剪枝权重参数，在输入维度(dim=1)上进行剪枝
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
        # 更新模块的输入通道数
        self.module.in_channels = self.module.in_channels - len(pruned_idxes)

    def compute_flops(self, input_tensor_shape):
        """
        计算卷积操作的浮点运算次数(FLOPs)。
        Args:
            input_tensor_shape (tuple): 输入张量的形状，格式为[batch_size, in_channels, height, width]。
        Returns:
            int: 估计的FLOPs数量。
        注意:
            - 此计算仅考虑乘法操作
            - 计算公式考虑了批大小、卷积核尺寸、步长、滑动次数和通道数
        """

        # 添加输入形状检查
        if len(input_tensor_shape) < 4:
            print(f"警告: 算子 {self.id} 的输入张量形状 {input_tensor_shape} 不是4维，无法计算FLOPs")
            return 0  # 返回0表示无法计算

        # 解析输入张量形状
        batch_size, _, height_in, width_in = input_tensor_shape

        # 获取步长和卷积核尺寸
        stride_h, stride_w = self.cfg_params['strides']
        kernel_h, kernel_w = self.cfg_params['kernel_shape']
        
        # 如果配置中包含填充信息，调整输入高宽
        if 'pads' in self.cfg_params:
            height_in = height_in + self.cfg_params['pads'][2] * 2
            width_in = width_in + self.cfg_params['pads'][3] * 2
        # 计算卷积核在输入上滑动的次数
        sliding_times_h = (height_in - kernel_h + stride_h) // (stride_h)
        sliding_times_w = (width_in - kernel_w + stride_w) // (stride_w)
        # 计算总FLOPs：批大小 * 卷积核元素数 * 滑动次数 * 输入通道数 * 输出通道数
        flops = batch_size * kernel_h * kernel_w * sliding_times_h * sliding_times_w \
                * self.module.in_channels * self.module.out_channels
        # 如果是分组卷积，FLOPs需要除以组数
        if 'group' in self.cfg_params:
            flops /= self.cfg_params['group'] 
        return flops
    
class ConvTranspose2dOTO(Operator):
    """
    反卷积2D算子类，封装并扩展了PyTorch的nn.ConvTranspose2d模块。
    提供了参数管理、通道维度剪枝和参数分组等功能。
    注意：反卷积的权重形状与标准卷积不同，剪枝维度需要相应调整。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化ConvTranspose2dOTO算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为None。
            _type (str, optional): 算子的类型名称。默认为None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.ConvTranspose2d, optional): 关联的PyTorch反卷积模块。默认为None。
        """
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = True         # 标记为Stem算子，表示可以转换张量的主维度
        self.set_num_groups()       # 设置分组数，用于剪枝或变换
        # 设置参数变换类型为TRANSPOSE，指示权重需要特殊处理（例如，剪枝维度不同）
        self.p_transform = TensorTransform.TRANSPOSE
        
    def set_num_groups(self):
        """
        设置算子的num_groups属性。
        对于反卷积，通常基于参数的形状来确定分组数量。
        权重形状为 (in_channels, out_channels // groups, kH, kW)。
        偏置形状为 (out_channels,)。
        这里将 num_groups 设置为权重第二维（输出通道相关）和偏置第一维的最大值。
        """
        self.num_groups = 1  # 初始化为 1
        for param_name in self.name_to_param:
            param = self.name_to_param[param_name]
            if param_name.endswith('.weight'):
                # 对于权重，使用第二维的大小 (out_channels // groups) 作为分组依据
                self.num_groups = max(self.num_groups, param.shape[1])
            elif param_name.endswith('.bias'):
                # 对于偏置，使用第一维的大小 (out_channels) 作为分组依据
                self.num_groups = max(self.num_groups, param.shape[0])
            
    def get_param_groups(self, param_names=[]):
        """
        获取反卷积算子的参数分组信息，用于优化器配置。
        Args:
            param_names (list, optional): 要包含在参数组中的参数名称列表。默认为空列表。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()
        param_groups['op'] = 'convtranspose2d'      # 设置操作类型为反卷积2D
        param_groups['p_names'] = list()            # 初始化参数名称列表
        param_groups['params'] = list()             # 初始化参数对象列表
        param_groups['p_transform'] = list()        # 初始化参数变换类型列表
        # 遍历指定的参数名称，收集对应的参数和变换类型
        for p_name in param_names:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            # 使用为该算子设置的特定变换类型 (TRANSPOSE)
            param_groups['p_transform'].append(self.p_transform)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝反卷积算子的输出通道维度 (module.out_channels)。
        这对应于权重张量的第二维 (dim=1) 和偏置张量的第一维 (dim=0)。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输出通道索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        注意:
            - 代码中使用了 self.module.transposed 判断，但 nn.ConvTranspose2d 没有此属性。
              正确的剪枝维度对于权重是 dim=1，对于偏置是 dim=0。
            - 此处保留了原始代码逻辑，但注释指出了正确的维度。
        """
        # 计算需要保留的输出通道索引
        preserved_idxes = list(set(range(self.module.out_channels)) - set(pruned_idxes))
        preserved_idxes.sort()  # 对索引排序
        # 更新模块的输出通道数
        self.module.out_channels = self.module.out_channels - len(pruned_idxes)

        # 剪枝权重参数
        # 原始代码逻辑: 根据一个不存在的 transposed 标志选择剪枝维度
        # 正确逻辑: 对于 out_channels，应剪枝权重的 dim=1
        if not self.module.transposed:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
        else:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
        
        if self.module.bias is not None:
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)
        
    def prune_in_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝反卷积算子的输入通道维度 (module.in_channels)。
        这对应于权重张量的第一维 (dim=0)。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输入通道索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        注意:
            - 代码中使用了 self.module.transposed 判断，但 nn.ConvTranspose2d 没有此属性。
              正确的剪枝维度对于权重是 dim=0。
            - 此处保留了原始代码逻辑，但注释指出了正确的维度。
        """
        # 计算需要保留的输入通道索引
        preserved_idxes = list(set(range(self.module.in_channels)) - set(pruned_idxes))
        preserved_idxes.sort()  # 对索引排序
        # 更新模块的输入通道数
        self.module.in_channels = self.module.in_channels - len(pruned_idxes)
        
        # 剪枝权重参数
        # 原始代码逻辑: 根据一个不存在的 transposed 标志选择剪枝维度
        # 正确逻辑: 对于 in_channels，应剪枝权重的 dim=0
        if not self.module.transposed:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
        else:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)

            
class BatchNormOTO(Operator):
    """
    批归一化算子类，封装并扩展了PyTorch的nn.BatchNorm2d (或类似) 模块。
    提供了参数管理、特征维度剪枝和参数分组等功能。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化BatchNormOTO算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为None。
            _type (str, optional): 算子的类型名称。默认为None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.BatchNorm2d, optional): 关联的PyTorch批归一化模块。默认为None。
        """
        super().__init__(id, _type, cfg_params, module)         # 调用父类 Operator 的初始化方法
        self.is_stem = False                                    # 标记为非 Stem 算子，BatchNorm 不改变通道数（特征数）
        self.set_num_groups()                                   # 设置分组数，对于 BatchNorm，通常是特征数 (num_features)
        self.p_transform = TensorTransform.ACCESSORY            # 设置参数变换类型为 ACCESSORY，表示这些参数（如 gamma, beta）通常与主要权重（如卷积核）一起剪枝
        
    def get_param_groups(self, param_names=[]):
        """
        获取批归一化算子的参数分组信息，用于优化器配置。
        Args:
            param_names (list, optional): 要包含在参数组中的参数名称列表。默认为空列表。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                   # 初始化参数组字典
        param_groups['op'] = 'batchnorm'        # 设置操作类型为 'batchnorm'
        param_groups['p_names'] = list()        # 初始化参数名称列表
        param_groups['params'] = list()         # 初始化参数对象列表
        param_groups['p_transform'] = list()    # 初始化参数变换类型列表
        # 遍历指定的参数名称
        for p_name in param_names:
            # 检查参数名称是否存在于算子的参数映射中
            if p_name in self.name_to_param:
                param_groups['p_names'].append(p_name)                      # 将参数名称添加到列表
                param_groups['params'].append(self.name_to_param[p_name])   # 将参数对象添加到列表
                param_groups['p_transform'].append(self.p_transform)        # 将参数变换类型添加到列表
        # 返回参数组字典
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝批归一化算子的输出维度（特征维度）。
        这会影响 num_features 属性以及相关的 buffers (running_mean, running_var)
        和可学习参数 (weight, bias)。
        Args:
            pruned_idxes (list, optional): 需要剪枝的特征索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        """
        preserved_idxes = list(set(range(self.module.num_features)) - set(pruned_idxes))    # 计算需要保留的特征索引
        preserved_idxes.sort()                                                              # 对保留的索引进行排序，保持相对顺序
        self.module.num_features = self.module.num_features - len(pruned_idxes)             # 更新模块的特征数量
        # 剪枝运行均值 (running_mean)，这是一个 buffer，不是 parameter，直接索引 .data
        self.module.running_mean = self.module.running_mean.data[preserved_idxes]
        # 剪枝运行方差 (running_var)，这也是一个 buffer，直接索引 .data
        self.module.running_var = self.module.running_var.data[preserved_idxes]
        # 如果 BatchNorm 层包含可学习的仿射参数 (gamma 和 beta)
        if self.module.affine:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)      # 剪枝可学习的权重 (gamma) 参数及其梯度
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)          # 剪枝可学习的偏置 (beta) 参数及其梯度

class InstanceNormOTO(Operator):
    """
    实例归一化算子类，封装并扩展了PyTorch的nn.InstanceNorm (1d, 2d, 3d) 模块。
    提供了参数管理、特征维度剪枝和参数分组等功能。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化InstanceNormOTO算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为None。
            _type (str, optional): 算子的类型名称。默认为None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.InstanceNorm, optional): 关联的PyTorch实例归一化模块。默认为None。
        """
        super().__init__(id, _type, cfg_params, module)     # 调用父类 Operator 的初始化方法
        self.is_stem = False                                # 标记为非 Stem 算子，InstanceNorm 不改变通道数（特征数）
        self.set_num_groups()                               # 设置分组数，对于 InstanceNorm，通常是特征数 (num_features)
        self.p_transform = TensorTransform.ACCESSORY        # 设置参数变换类型为 ACCESSORY，表示这些参数（如 gamma, beta）通常与主要权重一起剪枝

    def get_param_groups(self, param_names=[]):
        """
        获取实例归一化算子的参数分组信息，用于优化器配置。
        Args:
            param_names (list, optional): 要包含在参数组中的参数名称列表。默认为空列表。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                       # 初始化参数组字典
        param_groups['op'] = 'instantnorm'          # 设置操作类型为 'instantnorm' (注意：可能是笔误，通常应为 'instancenorm')
        param_groups['p_names'] = list()            # 初始化参数名称列表
        param_groups['params'] = list()             # 初始化参数对象列表
        param_groups['p_transform'] = list()        # 初始化参数变换类型列表
        # 遍历指定的参数名称
        for p_name in param_names:
            param_groups['p_names'].append(p_name)                          # 将参数名称添加到列表
            param_groups['params'].append(self.name_to_param[p_name])       # 将参数对象添加到列表
            param_groups['p_transform'].append(self.p_transform)            # 将参数变换类型添加到列表
        return param_groups                                                 # 返回参数组字典

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝实例归一化算子的输出维度（特征维度）。
        这会影响 num_features 属性以及可学习参数 (weight, bias)（如果存在）。
        InstanceNorm 没有 running_mean 和 running_var buffers。
        Args:
            pruned_idxes (list, optional): 需要剪枝的特征索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        """
        preserved_idxes = list(set(range(self.module.num_features)) - set(pruned_idxes)) # 计算需要保留的特征索引
        preserved_idxes.sort()# 对保留的索引进行排序，保持相对顺序
        self.module.num_features = self.module.num_features - len(pruned_idxes)# 更新模块的特征数量
        # 如果 InstanceNorm 层包含可学习的仿射参数 (gamma 和 beta)
        if self.module.affine:
            # 剪枝可学习的权重 (gamma) 参数及其梯度
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
            # 剪枝可学习的偏置 (beta) 参数及其梯度
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)

class GroupNormOTO(Operator):
    """
    分组归一化算子类，封装并扩展了PyTorch的nn.GroupNorm模块。
    提供了参数管理、特征维度剪枝（按通道）和参数分组等功能。
    特别地，它引入了类似多头注意力的分组概念（num_heads, head_dim）用于参数变换和剪枝。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化GroupNormOTO算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为None。
            _type (str, optional): 算子的类型名称。默认为None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.GroupNorm, optional): 关联的PyTorch GroupNorm模块。默认为None。
        """
        super().__init__(id, _type, cfg_params, module)     # 调用父类 Operator 的初始化方法
        self.is_stem = False                                # 标记为非 Stem 算子，GroupNorm 不改变通道数
        self.set_num_groups()                               # 调用父类方法设置 num_groups，默认基于参数 shape[0]，对于 GroupNorm 可能是 num_channels
        # 将 PyTorch GroupNorm 模块的 num_groups 存储为 num_heads
        # 注意：这里的命名可能引起混淆，num_heads 实际上是 GroupNorm 的分组数
        self.num_heads = module.num_groups
        # 计算每个组（头）的维度（通道数 / 组数）
        # self.num_groups 在这里可能指代 num_channels (由 set_num_groups() 设置)
        self.head_dim = self.num_groups // self.num_heads
        # 再次将 self.num_groups 设置为模块的 num_groups。这可能覆盖了之前的 num_channels 值。
        # 这里的意图可能是明确区分通道数和 GroupNorm 的分组数，但命名不够清晰。
        self.num_groups = module.num_groups
        # 设置参数变换类型为MULTIHEAD_NUMHEAD_SPREAD，用于按“头”（即 GroupNorm 的组）进行变换或剪枝
        self.p_transform = TensorTransform.MULTIHEAD_NUMHEAD_SPREAD
        
    def get_param_groups(self, param_names=list()):
        """
        获取分组归一化算子的参数分组信息，用于优化器配置。
        除了标准的参数信息，还包含了 num_groups, num_heads, head_dim 等用于特定变换。
        Args:
            param_names (list, optional): 要包含在参数组中的参数名称列表。默认为空列表。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                               # 初始化参数组字典
        param_groups['op'] = 'groupnorm'                    # 设置操作类型为 'groupnorm'
        param_groups['p_names'] = list()                    # 初始化参数名称列表
        param_groups['params'] = list()                     # 初始化参数对象列表
        param_groups['p_transform'] = list()                # 初始化参数变换类型列表
        param_groups['num_groups'] = self.num_groups        # 添加 GroupNorm 的分组数
        param_groups['num_heads'] = self.num_heads          # 添加“头”数（即 GroupNorm 的分组数）
        param_groups['head_dim'] = self.head_dim            # 添加每个“头”的维度
        # 遍历指定的参数名称
        for p_name in param_names:
            param_groups['p_names'].append(p_name)                          # 将参数名称添加到列表
            param_groups['params'].append(self.name_to_param[p_name])       # 将参数对象添加到列表
            param_groups['p_transform'].append(self.p_transform)            # 将参数变换类型添加到列表
        return param_groups                                                 # 返回参数组字典

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝分组归一化算子的输出维度（通道维度）。
        这会影响num_channels 属性、num_groups属性以及可学习参数 (weight, bias)。
        Args:
            pruned_idxes (list, optional): 需要剪枝的通道索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        注意:
            - 假设通道的剪枝是按head_dim对齐的，即每次剪枝会移除整数个组。
        """
        preserved_idxes = list(set(range(self.module.num_channels)) - set(pruned_idxes))                # 计算需要保留的通道索引
        preserved_idxes.sort()                                                                          # 对保留的索引进行排序
        self.module.num_channels = self.module.num_channels - len(pruned_idxes)                         # 更新模块的通道数
        # 更新模块的分组数，假设剪枝的通道数是 head_dim 的整数倍
        self.module.num_groups = self.module.num_groups - len(pruned_idxes) // self.head_dim
        # 如果 GroupNorm 层包含可学习的仿射参数 (gamma 和 beta)
        if self.module.affine:
            self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)       # 剪枝可学习的权重 (gamma) 参数及其梯度
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)           # 剪枝可学习的偏置 (beta) 参数及其梯度


class LinearOTO(Operator):
    """
    线性（全连接）层算子类，封装并扩展了PyTorch的nn.Linear模块。
    提供了参数管理、输入/输出特征维度剪枝和计算量估计等功能。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化LinearOTO算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为None。
            _type (str, optional): 算子的类型名称。默认为None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.Linear, optional): 关联的PyTorch线性层模块。默认为None。
        """
        # 调用父类 Operator 的初始化方法
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = True             # 标记为 Stem 算子，表示可以转换输入张量的主维度（特征维度）
        self.set_num_groups()           # 设置分组数，对于线性层，通常是输出特征数 (out_features)
    
    def get_param_groups(self, param_names=list()):
        """
        获取线性层算子的参数分组信息，用于优化器配置。
        Args:
            param_names (list, optional): 要包含在参数组中的参数名称列表。如果为空，则默认包含所有参数。默认为空列表。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                   # 初始化参数组字典
        param_groups['op'] = 'linear'           # 设置操作类型为 'linear'
        param_groups['p_names'] = list()        # 初始化参数名称列表
        param_groups['params'] = list()         # 初始化参数对象列表
        param_groups['p_transform'] = list()    # 初始化参数变换类型列表
        # 确定目标参数名称：如果提供了 param_names 则使用它，否则使用算子所有的参数名称
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        # 遍历目标参数名称
        for p_name in target_param_names:
            param_groups['p_names'].append(p_name)                          # 将参数名称添加到列表
            param_groups['params'].append(self.name_to_param[p_name])       # 将参数对象添加到列表
            param_groups['p_transform'].append(self.p_transform)            # 将参数变换类型添加到列表
        return param_groups                                                 # 返回参数组字典

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝线性层的输出特征维度。
        这会影响 out_features 属性以及权重和偏置参数的第 0 维。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输出特征索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        """
        preserved_idxes = list(set(range(self.module.out_features)) - set(pruned_idxes))            # 计算需要保留的输出特征索引
        preserved_idxes.sort()                                                                      # 对保留的索引进行排序
        self.module.out_features = self.module.out_features - len(pruned_idxes)                     # 更新模块的输出特征数
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)      # 剪枝权重参数的输出维度 (dim=0)
        # 如果存在偏置，也对其进行剪枝 (dim=0)
        if self.module.bias is not None:
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)

    def prune_in_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝线性层的输入特征维度。
        这会影响 in_features 属性以及权重参数的第 1 维。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输入特征索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        """
        preserved_idxes = list(set(range(self.module.in_features)) - set(pruned_idxes))                 # 计算需要保留的输入特征索引
        preserved_idxes.sort()                                                                          # 对保留的索引进行排序
        self.module.in_features = self.module.in_features - len(pruned_idxes)                           # 更新模块的输入特征数
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)          # 剪枝权重参数的输入维度 (dim=1)

    def compute_flops(self, input_tensor_shape):
        # Only consider multiplication
        # The input_tensor_shape for linear is [*, in_features]
        flops = 1
        for dim in input_tensor_shape:
            flops *= dim
        flops *= self.module.out_features
        return flops

class LoraLinearOTO(Operator):
    """
    LoRA (Low-Rank Adaptation) 线性层算子类。
    封装了应用了 LoRA 的线性层模块 (通常来自 peft 库或类似实现)。
    提供了参数管理、维度剪枝（区分原始权重和 LoRA 矩阵）和参数分组等功能。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化 LoraLinearOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (LoraLinear module, optional): 关联的 LoRA 线性层模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params, module)         # 调用父类 Operator 的初始化方法
        self.set_num_groups()                                   # 设置分组数，通常基于父类的默认实现（可能基于输出维度）
        self.ori_in_features = self.module.in_features          # 存储原始（未剪枝前）的输入特征数
        self.ori_out_features = self.module.out_features        # 存储原始（未剪枝前）的输出特征数
        self.lora_scaling = module.scaling                      # 存储 LoRA 的缩放因子 alpha / r
        self.is_stem = True                                     # 标记为 Stem 算子，表示可以转换主维度（特征维度）
        self.is_basic = False                                   # 标记为非基础模块（因为包含 LoRA 结构）

    def get_param_groups(self, param_names=list(), skip_output_node=False, **kwargs):
        """
        获取 LoRA 线性层算子的参数分组信息。
        特殊处理 LoRA 参数 (lora_A, lora_B) 和原始权重/偏置。
        'lora_A' 参数通常不被剪枝 (NO_PRUNE)。
        Args:
            param_names (list, optional): 要包含在参数组中的参数名称列表。如果为空，则默认包含所有参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出节点（通常指 lora_B 和原始权重/偏置）。默认为 False。
            **kwargs: 其他可选参数。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                       # 初始化参数组字典
        param_groups['op'] = 'lora_linear'          # 设置操作类型为 'lora_linear'
        param_groups['p_names'] = list()            # 初始化参数名称列表
        param_groups['params'] = list()             # 初始化参数对象列表
        param_groups['p_transform'] = list()        # 初始化参数变换类型列表
        # 如果存在 LoRA 缩放因子，添加到参数组信息中
        if hasattr(self, 'lora_scaling'):
            param_groups['lora_scaling'] = self.lora_scaling
        # 确定目标参数名称：如果提供了 param_names 则使用它，否则使用算子所有的参数名称
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        # 遍历目标参数名称
        for p_name in target_param_names:
            param = self.name_to_param[p_name]      # 获取参数对象
            # 根据 skip_output_node 决定是否包含该参数
            # 如果不跳过输出节点，或者跳过但当前参数是 lora_A，则包含
            if not skip_output_node or (skip_output_node and 'lora_A' in p_name):
                param_groups['p_names'].append(p_name)  # 添加参数名称
                param_groups['params'].append(param)    # 添加参数对象
                # 为 lora_A 设置不剪枝变换类型
                if 'lora_A' in p_name:          
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                # 为其他参数（lora_B, weight, bias）设置基础变换类型
                else:
                    param_groups['p_transform'].append(TensorTransform.BASIC)
        # 返回参数组字典
        return param_groups
        
        
    def prune_out_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=False, **kwargs):  
        """
        剪枝 LoRA 线性层的输出维度。
        会影响原始权重/偏置的第 0 维和 lora_B 权重的第 0 维。
        lora_A 的输出维度（秩 r）通常不被此方法剪枝。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输出特征索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表（剪枝所有相关参数）。
            skip_output_node (bool, optional): (在此方法中未使用)。
            **kwargs: 其他可选参数。
        """
        # 如果提供了 param_names，则只剪枝指定的参数，否则剪枝所有参数
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param
        # 遍历目标参数名称
        for param_name in target_param_names:
            # 计算需要保留的输出特征索引
            # 注意：这里使用了 self.num_groups，它可能基于原始输出维度设置
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))
            preserved_idxes.sort() # 对索引排序
            # 处理原始权重和偏置
            if 'lora_A' not in param_name and 'lora_B' not in param_name:
                # 如果是原始权重
                if param_name.endswith('.weight'):
                    # 剪枝权重的输出维度 (dim=0)
                    self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
                    # 更新 name_to_param 字典中的引用
                    self.name_to_param[param_name] = self.module.weight
                # 如果是原始偏置
                elif param_name.endswith('.bias'):
                    # 剪枝偏置的维度 (dim=0)
                    self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)
                    # 更新 name_to_param 字典中的引用
                    self.name_to_param[param_name] = self.module.bias
                # 更新模块的输出特征数
                self.module.out_features = len(preserved_idxes)
            # 处理 lora_B 权重
            elif 'lora_B' in param_name:
                # LoRA 可能有多个 adapter，遍历 lora_B 中的所有模块
                for module in self.module.lora_B.values():
                    # 剪枝 lora_B 权重的输出维度 (dim=0)
                    module.weight = self.prune_param_and_grad(module.weight, preserved_idxes, 0)
                    # 更新 name_to_param 字典中的引用 (注意：这里可能只更新了最后一个 adapter 的引用)
                    self.name_to_param[param_name] = module.weight
                    module.out_features = len(preserved_idxes)      # 更新 lora_B 模块的输出特征数
                self.module.out_features = len(preserved_idxes)     # 更新主模块的输出特征数

    def prune_in_dim(self, pruned_idxes=list(), param_names=list(), verbose=False, **kwargs):
        """
        剪枝 LoRA 线性层的输入维度。
        会影响原始权重的第 1 维和 lora_A 权重的第 1 维。
        lora_B 的输入维度（秩 r）通常不被此方法剪枝。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输入特征索引列表。默认为空列表。
            param_names (list, optional): 必须提供，指定要剪枝的参数名称。
            verbose (bool, optional): 是否打印详细信息（未使用）。
            **kwargs: 其他可选参数。
        """
        # 遍历指定的参数名称
        for param_name in param_names:
            # 计算需要保留的输入特征索引，基于原始输入特征数
            preserved_idxes = list(set(range(self.ori_in_features)) - set(pruned_idxes))
            preserved_idxes.sort() # 对索引排序
            
            # 处理原始权重
            if 'lora_A' not in param_name and 'lora_B' not in param_name:
                # 如果是原始权重
                if param_name.endswith('.weight'):
                    # 剪枝权重的输入维度 (dim=1)
                    self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
                    # 更新 name_to_param 字典中的引用
                    self.name_to_param[param_name] = self.module.weight
                    # 更新模块的输入特征数
                    self.module.in_features = len(preserved_idxes)
            # lora_B 的输入维度是 LoRA rank (r)，不在此处剪枝
            elif 'lora_B' in param_name:
                pass # 不处理 lora_B 的输入维度
            # 处理 lora_A 权重
            elif 'lora_A' in param_name:
                # LoRA 可能有多个 adapter，遍历 lora_A 中的所有模块
                for module in self.module.lora_A.values():
                    # 剪枝 lora_A 权重的输入维度 (dim=1)
                    module.weight = self.prune_param_and_grad(module.weight, preserved_idxes, 1)
                    # 更新 name_to_param 字典中的引用 (注意：这里可能只更新了最后一个 adapter 的引用)
                    self.name_to_param[param_name] = module.weight
                    # 更新 lora_A 模块的输入特征数
                    module.in_features = len(preserved_idxes)
                # 更新主模块的输入特征数
                self.module.in_features = len(preserved_idxes)

class LoraEmbeddingOTO(Operator):
    '''
    LoRA (Low-Rank Adaptation) 嵌入层算子类。
    封装了应用了 LoRA 的嵌入层模块 (通常来自 peft 库或类似实现)。
    提供了参数管理、维度剪枝（区分原始嵌入权重和 LoRA 矩阵）和参数分组等功能。
    示例参数形状:
    model.model.embed_tokens.weight torch.Size([32000, 4096])                   # 原始嵌入权重 (vocab_size, embedding_dim)
    model.model.embed_tokens.lora_embedding_A.default torch.Size([8, 32000])    # LoRA A 矩阵 (rank, vocab_size)
    model.model.embed_tokens.lora_embedding_B.default torch.Size([4096, 8])     # LoRA B 矩阵 (embedding_dim, rank)
    '''
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化 LoraEmbeddingOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (LoraEmbedding module, optional): 关联的 LoRA 嵌入层模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params, module)             # 调用父类 Operator 的初始化方法
        self.num_groups = self.module.embedding_dim                 # 设置分组数，对于嵌入层，通常是嵌入维度 (embedding_dim)
        self.is_stem = True                                         # 标记为 Stem 算子，表示可以转换主维度（嵌入维度）
        self.is_basic = False                                       # 标记为非基础模块（因为包含 LoRA 结构）
        self.lora_scaling = module.scaling                          # 存储 LoRA 的缩放因子 alpha / r

    def get_param_groups(self, param_names=list(), skip_output_node=False, **kwargs):
        """
        获取 LoRA 嵌入层算子的参数分组信息。
        特殊处理 LoRA 参数 (lora_embedding_A, lora_embedding_B) 和原始嵌入权重。
        'lora_embedding_A' 参数通常不被剪枝 (NO_PRUNE)。
        原始嵌入权重需要转置处理 (TRANSPOSE)。
        Args:
            param_names (list, optional): 要包含在参数组中的参数名称列表。如果为空，则默认包含所有参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出节点（通常指 lora_B 和原始权重）。默认为 False。
            **kwargs: 其他可选参数。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                       # 初始化参数组字典
        param_groups['op'] = 'lora_embedding'       # 设置操作类型为 'lora_embedding'
        param_groups['p_names'] = list()            # 初始化参数名称列表
        param_groups['params'] = list()             # 初始化参数对象列表
        param_groups['p_transform'] = list()        # 初始化参数变换类型列表
        # 如果存在 LoRA 缩放因子，添加到参数组信息中
        if hasattr(self, 'lora_scaling'):
            param_groups['lora_scaling'] = self.lora_scaling
        # 确定目标参数名称：如果提供了 param_names 则使用它，否则使用算子所有的参数名称
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        # 遍历目标参数名称
        for p_name in target_param_names:
            param = self.name_to_param[p_name]      # 获取参数对象
            # 根据 skip_output_node 决定是否包含该参数
            # 如果不跳过输出节点，或者跳过但当前参数是 lora_embedding_A，则包含
            if not skip_output_node or (skip_output_node and 'lora_embedding_A' in p_name):
                param_groups['p_names'].append(p_name)      # 添加参数名称
                param_groups['params'].append(param)        # 添加参数对象
                # 为 lora_embedding_A 设置不剪枝变换类型
                if 'lora_embedding_A' in p_name:          
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                # 为 lora_embedding_B 设置基础变换类型
                elif 'lora_embedding_B' in p_name:
                    param_groups['p_transform'].append(TensorTransform.BASIC)
                # 为原始嵌入权重设置转置变换类型
                else:
                    param_groups['p_transform'].append(TensorTransform.TRANSPOSE)
        return param_groups                                 # 返回参数组字典
    
    def prune_out_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=False, **kwargs):  
        """
        剪枝 LoRA 嵌入层的输出维度（embedding_dim）。
        会影响原始嵌入权重的第 1 维和 lora_embedding_B 权重的第 0 维。
        lora_embedding_A 的维度（词表大小 vocab_size）通常不被此方法剪枝。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输出维度索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表（剪枝所有相关参数）。
            skip_output_node (bool, optional): (在此方法中未使用)。
            **kwargs: 其他可选参数。
        """
        preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))         # 计算需要保留的输出维度索引
        preserved_idxes.sort()                                                          # 对索引排序
        # 如果提供了 param_names，则只剪枝指定的参数，否则剪枝所有参数
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param
        # 遍历目标参数名称
        for param_name in target_param_names:   
            # 处理原始嵌入权重
            if 'lora_embedding_A' not in param_name and 'lora_embedding_B' not in param_name:
                # 如果是原始权重
                if param_name.endswith('.weight'):
                    # 剪枝权重的嵌入维度 (dim=1)
                    self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)
                    # 更新 name_to_param 字典中的引用
                    self.name_to_param[param_name] = self.module.weight
                # 更新模块的嵌入维度
                self.module.embedding_dim = len(preserved_idxes)
            # 处理 lora_embedding_B 权重
            elif 'lora_embedding_B' in param_name:
                # LoRA 可能有多个 adapter，遍历 lora_embedding_B 中的所有模块参数
                for module_name in self.module.lora_embedding_B:
                    module_param = self.module.lora_embedding_B[module_name]            # 获取当前 adapter 的参数
                    # 剪枝 lora_embedding_B 权重的输出维度 (dim=0)
                    self.module.lora_embedding_B[module_name] = self.prune_param_and_grad(module_param, preserved_idxes, 0)
                    # 更新 name_to_param 字典中的引用 (注意：这里可能只更新了最后一个 adapter 的引用)
                    self.name_to_param[param_name] = module_param # 修正：应该更新为剪枝后的参数
                    # 注意：原始代码中 self.name_to_param[param_name] = module_param，这可能导致引用未更新
                    # 建议改为：self.name_to_param[param_name] = self.module.lora_embedding_B[module_name]
                self.module.embedding_dim = len(preserved_idxes)                        # 更新主模块的嵌入维度

class EmbeddingOTO(Operator):
    """
    标准嵌入层算子类，封装并扩展了PyTorch的nn.Embedding模块。
    提供了参数管理、嵌入维度剪枝和参数分组等功能。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化 EmbeddingOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.Embedding, optional): 关联的 PyTorch 嵌入层模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params, module)         # 调用父类 Operator 的初始化方法
        self.num_groups = self.module.embedding_dim             # 设置分组数，对于嵌入层，通常是嵌入维度 (embedding_dim)
        # 标记权重参数是否需要转置处理（在某些剪枝或变换场景下）
        # 对于 Embedding，权重形状是 (num_embeddings, embedding_dim)，如果将 embedding_dim 视为“输出”维度进行剪枝，则需要处理 dim=1，
        self.is_transpose = True                                # 这类似于转置后的卷积权重，因此标记为 True
        self.is_stem = False                                    # 标记为非 Stem 算子，嵌入层通常不直接连接输入，或者其主要维度（词表大小）不参与通道剪枝
        self.is_basic = True                                    # 标记为基础模块
        self.p_transform = TensorTransform.TRANSPOSE            # 设置参数变换类型为 TRANSPOSE，指示权重需要特殊处理（剪枝 dim=1）
        
    def get_param_groups(self, **kwargs):
        """
        获取嵌入层算子的参数分组信息，用于优化器配置。
        Args:
            **kwargs: 其他可选参数。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                                               # 初始化参数组字典
        param_groups['op'] = 'embedding'                                    # 设置操作类型为 'embedding'
        param_groups['p_names'] = list()                                    # 初始化参数名称列表
        param_groups['params'] = list()                                     # 初始化参数对象列表
        param_groups['p_transform'] = list()                                # 初始化参数变换类型列表
        # 遍历算子所有的参数名称
        for p_name in self.name_to_param:
            param_groups['p_names'].append(p_name)                          # 将参数名称添加到列表
            param_groups['params'].append(self.name_to_param[p_name])       # 将参数对象添加到列表
            param_groups['p_transform'].append(self.p_transform)            # 将参数变换类型 (TRANSPOSE) 添加到列表
        return param_groups                                                 # 返回参数组字典

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝嵌入层的输出维度（embedding_dim）。
        这会影响 embedding_dim 属性以及权重参数的第 1 维。
        Args:
            pruned_idxes (list, optional): 需要剪枝的嵌入维度索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        """
        preserved_idxes = list(set(range(self.module.embedding_dim)) - set(pruned_idxes))       # 计算需要保留的嵌入维度索引
        preserved_idxes.sort()                                                                  # 对保留的索引进行排序
        self.module.embedding_dim = self.module.embedding_dim - len(pruned_idxes)               # 更新模块的嵌入维度
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 1)  # 剪枝权重参数的嵌入维度 (dim=1)
    
class LayerNormOTO(Operator):
    """
    层归一化算子类，封装并扩展了PyTorch的nn.LayerNorm模块。
    提供了参数管理、归一化维度剪枝和参数分组等功能。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化 LayerNormOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.LayerNorm, optional): 关联的 PyTorch 层归一化模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params, module)     # 调用父类 Operator 的初始化方法
        self.set_num_groups()                               # 设置分组数，对于 LayerNorm，通常基于 normalized_shape 的最后一个维度
        self.is_stem = False                                # 标记为非 Stem 算子，LayerNorm 通常不改变特征/通道维度
        self.is_basic = False                               # 标记为非基础模块 (注意：这与其他归一化层如BatchNorm, InstanceNorm 不同，它们被标记为基础模块。这可能是特定设计选择或笔误)
            
    def get_param_groups(self, **kwargs):
        """
        获取层归一化算子的参数分组信息，用于优化器配置。
        Args:
            **kwargs: 其他可选参数。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                   # 初始化参数组字典
        param_groups['op'] = 'layernorm'        # 设置操作类型为 'layernorm'
        param_groups['p_names'] = list()        # 初始化参数名称列表
        param_groups['params'] = list()         # 初始化参数对象列表
        param_groups['p_transform'] = list()    # 初始化参数变换类型列表
        # 遍历算子所有的参数名称
        for p_name in self.name_to_param:
            param_groups['p_names'].append(p_name)                          # 将参数名称添加到列表
            param_groups['params'].append(self.name_to_param[p_name])       # 将参数对象添加到列表
            param_groups['p_transform'].append(self.p_transform)            # 将默认的参数变换类型添加到列表
        return param_groups                                                 # 返回参数组字典

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝层归一化算子的归一化维度。
        这会影响可学习参数 (weight, bias) 以及 normalized_shape 属性。
        Args:
            pruned_idxes (list, optional): 需要剪枝的维度索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        注意:
            - self.num_groups 在这里应该等于归一化维度的大小。
            - 假设 normalized_shape 是一个单元素元组。
        """
        # self.num_groups 应该等于原始的 normalized_shape[0]
        preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))                     # 计算需要保留的维度索引
        preserved_idxes.sort()                                                                      # 对保留的索引进行排序
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)      # 剪枝可学习的权重 (gamma) 参数及其梯度 (dim=0)
        # 如果存在可学习的偏置 (beta) 参数
        if hasattr(self.module, 'bias'):
            self.module.bias = self.prune_param_and_grad(self.module.bias, preserved_idxes, 0)      # 剪枝偏置参数及其梯度 (dim=0)
        # 如果模块具有 normalized_shape 属性
        if hasattr(self.module, 'normalized_shape'):
            # 更新 normalized_shape 属性以反映剪枝后的维度大小，假设原始 normalized_shape 是 (dim,) 的形式
            self.module.normalized_shape = tuple((len(preserved_idxes),))        


class ConditionOperatorOTO(Operator):
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        super().__init__(id, _type, cfg_params, module)
        self.is_stem = True
        self.num_heads = 2
        self.set_num_groups()
        self.head_dim = self.num_groups
        
    def set_num_groups(self):
        self.num_groups = 1e5
        for p_name in self.name_to_param:
            param = self.name_to_param[p_name]
            self.num_groups = min(self.num_groups, param.shape[0])

    def get_param_groups(self, **kwargs):
        param_groups = dict()
        param_groups['op'] = 'conditionOperator'
        param_groups['num_groups'] = self.num_groups
        param_groups['p_names'] = list()
        param_groups['params'] = list()
        param_groups['p_transform'] = list()
        param_groups['num_heads'] = self.num_heads
        param_groups['head_dim'] = self.head_dim
        for p_name in self.name_to_param:
            param_groups['p_names'].append(p_name)
            param_groups['params'].append(self.name_to_param[p_name])
            if 'cond_fc' in p_name:
                param_groups['p_transform'].append(TensorTransform.MULTIHEAD_HEADDIM)
            else:
                param_groups['p_transform'].append(TensorTransform.BASIC)
        return param_groups

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        for module_name in self.leaf_modules:
            leaf_op = self.leaf_modules[module_name]
            if len(leaf_op.param_names) == 0:
                continue
            if module_name.endswith('cond_fc.1'):
                refined_prune_idxes = []
                for i in range(2):
                    refined_prune_idxes += [p_i + i * self.num_groups for p_i in pruned_idxes]
                leaf_op.prune_out_dim(refined_prune_idxes)
            else:
                leaf_op.prune_out_dim(pruned_idxes)

    def prune_in_dim(self, pruned_idxes=list(), param_names=list(), **kwargs):
        visited_ops = set()
        if len(param_names) > 0:
            for param_name in param_names:
                for module_name in self.leaf_modules:
                    if not param_name.startswith(module_name):
                        continue
                    leaf_op = self.leaf_modules[module_name]
                    if leaf_op.id not in visited_ops:
                        leaf_op.prune_in_dim(pruned_idxes, param_names=[param_name])
                    visited_ops.add(leaf_op.id)


class BaseMultiHeadAttentionOTO(Operator):
    """
    多头注意力机制的基础算子类，封装了常见的 MHA 模块变体。
    提供了参数管理、基于头维度(head_dim)或头数量(num_head)的剪枝，以及参数分组功能。
    这是一个复合算子，其内部包含多个叶子模块（如线性层）。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None, model=None):
        """
        初始化 BaseMultiHeadAttentionOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.Module, optional): 关联的 PyTorch 多头注意力模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params, module) # 调用父类 Operator 的初始化方法
        self.root_model = model
        self.is_stem = True                             # 标记为 Stem 算子，表示可以转换主维度（隐藏层大小）
        self.is_basic = False                           # 标记为非基础模块（因为是复合模块）
        self.out_key = 'attn_w'                         # 默认的输出投影层（或相关参数）的名称关键字
        self.op_name = 'multi_head_attention'           # 算子名称
        self.prune_mode = 'head_dim'                    # 默认剪枝模式：按头的数量剪枝
        # 断言剪枝模式必须是 'head_dim' 或 'num_head'
        assert self.prune_mode in ['head_dim', 'num_head'], "Prune_mode of mha must be either head_dim or num_head."

        # 如果内部包含 LoRA 线性层，查找并存储 LoRA 缩放因子
        for leaf_module in self.leaf_modules.values():
            if type(leaf_module).__name__ == 'LoraLinearOTO':
                if hasattr(leaf_module, 'lora_scaling'):
                    self.lora_scaling = leaf_module.lora_scaling
        # 设置 MHA 相关的属性，如头数、头维度等
        self.set_attributes()
        
    def set_attributes(self):
        """
        根据关联的 PyTorch 模块设置 MHA 的核心属性。
        尝试从不同属性名（n_heads, num_heads）获取头数。
        根据剪枝模式设置 num_groups。
        """
        # 获取头数 (num_heads)
        if hasattr(self.module, 'n_heads'):
            self.num_heads = self.module.n_heads
        elif hasattr(self.module, 'num_heads'):
            self.num_heads = self.module.num_heads
        self.head_dim = self.module.head_dim            # 获取头维度 (head_dim)
        # 根据剪枝模式设置 num_groups
        if self.prune_mode == 'head_dim':
            self.num_groups = self.head_dim             # 按头维度剪枝时，分组数等于头维度
        elif self.prune_mode == 'num_head':
            self.num_groups = self.num_heads            # 按头数量剪枝时，分组数等于头数量
        self.hidden_size = self.module.hidden_size      # 获取隐藏层大小 (hidden_size)
        self.num_group_divisible = 2                    # 设置分组可除数，用于确保剪枝后的维度是有效的（例如，某些操作要求维度是偶数）
    
    def get_param_groups(self, param_names=list(), skip_output_node=False, **kwargs):
        """
        获取多头注意力算子的参数分组信息。
        特殊处理 LoRA 参数、输出投影层参数，并根据剪枝模式设置变换类型。
        Args:
            param_names (list, optional): 要包含在参数组中的参数名称列表。如果为空，则默认包含所有参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出投影相关的参数。默认为 False。
            **kwargs: 其他可选参数。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                       # 初始化参数组字典
        param_groups['op'] = self.op_name           # 设置操作类型
        param_groups['p_names'] = list()            # 初始化参数名称列表
        param_groups['params'] = list()             # 初始化参数对象列表
        param_groups['p_transform'] = list()        # 初始化参数变换类型列表
        param_groups['num_heads'] = self.num_heads  # 添加头数信息
        param_groups['head_dim'] = self.head_dim    # 添加头维度信息
        # 如果存在 LoRA 缩放因子，添加到参数组信息中
        if hasattr(self, 'lora_scaling'):
            param_groups['lora_scaling'] = self.lora_scaling
        # 确定目标参数名称
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        # 遍历目标参数名称
        for p_name in target_param_names:
            param = self.name_to_param[p_name]      # 获取参数对象
            # 处理输出投影层相关的参数
            if self.out_key in p_name and not skip_output_node:
                param_groups['p_names'].append(p_name)  # 添加参数名称
                param_groups['params'].append(param)    # 添加参数对象
                # 如果是 LoRA A 矩阵，设置为不剪枝
                if 'lora_A' in p_name:
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                # 否则，设置为基础变换类型
                else:
                    param_groups['p_transform'].append(TensorTransform.BASIC)
            # 处理非输出投影层的参数 (如 Q, K, V 投影)
            elif self.out_key not in p_name:
                param_groups['p_names'].append(p_name)  # 添加参数名称
                param_groups['params'].append(param)    # 添加参数对象
                # 如果是 LoRA A 矩阵，设置为不剪枝
                if 'lora_A' in p_name:
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                # 根据剪枝模式设置多头注意力相关的变换类型
                elif self.prune_mode == 'head_dim':
                    param_groups['p_transform'].append(TensorTransform.MULTIHEAD_HEADDIM)
                elif self.prune_mode == 'num_head':
                    param_groups['p_transform'].append(TensorTransform.MULTIHEAD_NUMHEAD)
        return param_groups                         # 返回参数组字典
    
    def prune_out_dim_head_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        """
        按头维度 (head_dim) 剪枝 MHA 的输出维度。
        这会影响 Q, K, V 投影层的输出维度以及模块的 head_dim 属性。
        输出投影层通常不在此处剪枝（由 skip_output_node 控制）。
        Args:
            pruned_idxes (list, optional): 需要在每个头内部剪枝的维度索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表（剪枝所有相关参数）。
            skip_output_node (bool, optional): 是否跳过输出投影层。默认为 True。
            **kwargs: 其他可选参数。
        """
        visited_modules = set()                                     # 记录已访问（处理过）的叶子模块
        # 情况一：提供了具体的参数名称列表
        if len(param_names) > 0:
            for param_name in param_names:
                # 遍历所有叶子模块
                for module_name in self.leaf_modules:
                    # 如果参数名称不属于当前叶子模块，则跳过
                    if not param_name.startswith(module_name):
                        continue
                    leaf_op = self.leaf_modules[module_name]        # 获取叶子算子
                    # 如果该叶子模块尚未处理过
                    if module_name not in visited_modules:
                        # 调用叶子算子的 prune_out_dim 方法，只处理当前参数
                        # 注意：这里传递的 pruned_idxes 是相对于单个头的维度索引
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    visited_modules.add(module_name) # 标记为已访问
        # 情况二：未提供参数名称列表，且跳过输出节点
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes))                 # 计算保留的头内维度索引
            preserved_idxes.sort()
            self.module.head_dim = self.module.head_dim - len(pruned_idxes)                         # 更新模块的头维度属性
            # 遍历所有叶子模块
            for module_name in self.leaf_modules:
                # 跳过输出投影层
                if self.out_key in module_name:
                    continue
                leaf_op = self.leaf_modules[module_name]                                            # 获取叶子算子
                expand_pruned_idxes = list()                                                        # 扩展剪枝索引到整个隐藏层维度
                # 对于每个头，将头内剪枝索引映射到全局索引
                for h in range(self.num_heads):
                    expand_pruned_idxes.extend([i + h * self.head_dim for i in pruned_idxes])       # 原始 head_dim
                leaf_op.prune_out_dim(expand_pruned_idxes)                                          # 调用叶子算子的 prune_out_dim 方法，使用扩展后的全局索引

    def prune_out_dim_num_head(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        """
        按头数量 (num_head) 剪枝 MHA 的输出维度。
        这会影响 Q, K, V 投影层的输出维度（移除整个头对应的部分）以及模块的头数属性。
        输出投影层通常不在此处剪枝（由 skip_output_node 控制）。
        Args:
            pruned_idxes (list, optional): 需要剪枝的头的索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表（剪枝所有相关参数）。
            skip_output_node (bool, optional): 是否跳过输出投影层。默认为 True。
            **kwargs: 其他可选参数。
        """
        visited_modules = set()   # 记录已访问（处理过）的叶子模块
        # 情况一：提供了具体的参数名称列表
        if len(param_names) > 0:
            for param_name in param_names:
                # 遍历所有叶子模块
                for module_name in self.leaf_modules:
                    # 如果参数名称不属于当前叶子模块，则跳过
                    if not param_name.startswith(module_name):
                        continue
                    leaf_op = self.leaf_modules[module_name]  # 获取叶子算子
                    # 如果该叶子模块尚未处理过
                    if module_name not in visited_modules:
                        # 调用叶子算子的 prune_out_dim 方法，只处理当前参数
                        # 注意：这里传递的 pruned_idxes 是头的索引
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    visited_modules.add(module_name) # 标记为已访问
        # 情况二：未提供参数名称列表，且跳过输出节点
        elif len(param_names) == 0 and skip_output_node:
            # 计算保留的头索引
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes)) # self.num_groups == self.num_heads
            preserved_idxes.sort()
            # 更新模块的头数属性
            if hasattr(self.module, 'n_heads'):
                self.module.n_heads = self.num_groups - len(pruned_idxes)
            elif hasattr(self.module, 'num_heads'):
                self.module.num_heads = self.num_groups - len(pruned_idxes)
                
            # 遍历所有叶子模块
            for module_name in self.leaf_modules:
                # 跳过输出投影层
                if self.out_key in module_name:
                    continue
                leaf_op = self.leaf_modules[module_name] # 获取叶子算子
                expand_pruned_idxes = list() # 扩展剪枝索引到整个隐藏层维度
                # 对于每个要剪枝的头索引 i
                for i in pruned_idxes:
                    # 将该头对应的所有维度索引添加到扩展列表中
                    for h in range(self.head_dim):
                        expand_pruned_idxes.append(h + i * self.head_dim)
                # 调用叶子算子的 prune_out_dim 方法，使用扩展后的全局索引
                leaf_op.prune_out_dim(expand_pruned_idxes)
                
    def prune_out_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        """
        根据设定的剪枝模式 (prune_mode) 调用相应的输出维度剪枝方法。
        Args:
            pruned_idxes (list, optional): 需要剪枝的索引列表（头维度或头数量）。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出投影层。默认为 True。
            **kwargs: 其他可选参数。
        """
        if self.prune_mode == 'head_dim':       
            self.prune_out_dim_head_dim(pruned_idxes, param_names, skip_output_node, **kwargs)
        elif self.prune_mode == 'num_head':
            self.prune_out_dim_num_head(pruned_idxes, param_names, skip_output_node, **kwargs)
        
    def prune_in_dim(self, pruned_idxes=list(), param_names=list(), **kwargs):
        """
        剪枝 MHA 的输入维度。
        将剪枝操作委托给所有相关的叶子模块（通常是 Q, K, V 和输出投影层）。
        Args:
            pruned_idxes (list, optional): 需要剪枝的输入维度索引列表。默认为空列表。
            param_names (list, optional): 必须提供，指定要剪枝的参数名称。
            **kwargs: 其他可选参数。
        """
        visited_modules = set() # 记录已访问（处理过）的叶子模块
        # 遍历指定的参数名称
        for param_name in param_names:
            # 遍历所有叶子模块
            for module_name in self.leaf_modules:
                # 如果参数名称不属于当前叶子模块，则跳过
                if not param_name.startswith(module_name):
                    continue
                leaf_op = self.leaf_modules[module_name] # 获取叶子算子
                # 如果叶子算子没有 prune_in_dim 方法，则跳过
                if not hasattr(leaf_op, 'prune_in_dim'):
                    continue
                # 如果该叶子模块尚未处理过
                if module_name not in visited_modules:
                    # 调用叶子算子的 prune_in_dim 方法，只处理当前参数
                    leaf_op.prune_in_dim(pruned_idxes, param_names=[param_name])
                visited_modules.add(module_name) # 标记为已访问

    """
    def compute_flops(self, input_tensor_shape):
        if isinstance(input_tensor_shape, (int, float)) or (isinstance(input_tensor_shape, (list, tuple)) and len(input_tensor_shape) == 1):
            # Scalar parameter or single-element tensor
            # we assume one operation (e.g., multiplication) per scalar
            return 1
        
        if not isinstance(input_tensor_shape, (list, tuple)):
            raise ValueError(f"Unexpected input_tensor_shape: {input_tensor_shape}")
        
        if len(input_tensor_shape) != 3:
            raise ValueError(f"Expected 3D input shape (batch_size, seq_len, hidden_size), got: {input_tensor_shape}")
        
        # flops for Q, K, V projections
        batch_size, seq_len, hidden_size = input_tensor_shape
        # flops = 3 * batch_size * seq_len * hidden_size * hidden_size
        # flops for attention scores and weights
        score_flops = batch_size * self.num_heads * seq_len * seq_len * self.head_dim
        weight_flops = batch_size * self.num_heads * seq_len * seq_len * self.head_dim
        # flops for output projection
        # output_flops = batch_size * seq_len * hidden_size * hidden_size
        # total_flops = flops + score_flops + weight_flops + output_flops
        total_flops = score_flops + weight_flops
        return total_flops

    def compute_macs(self, input_shape):
        # placholder
        return self.compute_flops(input_shape) // 2

    def compute_bops(self, input_tensor_shape, weight_bit=32, activation_bit=32):
        macs = self.compute_macs(input_tensor_shape)
        return macs * weight_bit * activation_bit
    """

class LlamaAttentionOTO(BaseMultiHeadAttentionOTO):
    """
    Llama 模型注意力机制的算子类，继承自 BaseMultiHeadAttentionOTO。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None, model=None):
        """
        初始化 LlamaAttentionOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (LlamaAttention, optional): 关联的 LlamaAttention 模块。默认为 None。
        """
        '''
        调试信息输出，帮助确认 LlamaAttention 模块的属性和配置。
        '''
        '''
        print(f"--- Debug LlamaAttentionOTO ---")
        print(f"Module type: {type(module)}")
        print(f"Module dir: {dir(module)}")
        if hasattr(module, 'config'):
            print(f"Module config type: {type(module.config)}")
            print(f"Module config dir: {dir(module.config)}")
            if hasattr(module.config, 'num_attention_heads'):
                print(f"Module config.num_attention_heads: {module.config.num_attention_heads}")
            if hasattr(module.config, 'num_key_value_heads'):
                print(f"Module config.num_key_value_heads: {module.config.num_key_value_heads}") # LLaMA 2 可能用这个
        if hasattr(module, 'num_heads'):
            print(f"Module has num_heads: {module.num_heads}")
        if hasattr(module, 'num_key_value_heads'): # LLaMA 2 可能直接在模块上有这个
             print(f"Module has num_key_value_heads: {module.num_key_value_heads}")
        print(f"--- End Debug LlamaAttentionOTO ---")
        '''
        # 从原始 LlamaAttention 模块获取注意力头的数量
        # 确保在调用父类构造函数之前设置 self.num_heads，因为父类的 set_attributes 方法会用到它。
        if hasattr(module, 'num_heads'):
            self.num_heads = module.num_heads
        elif hasattr(module, 'config') and hasattr(module.config, 'num_attention_heads'):
            self.num_heads = module.config.num_attention_heads
        # 尝试 LLaMA 2 中可能使用的 num_key_value_heads (如果 num_heads 和 num_attention_heads 都没有)
        # 对于分组查询注意力 (GQA) 或多查询注意力 (MQA)，num_heads (查询头) 可能与 num_key_value_heads 不同。
        elif hasattr(module, 'num_key_value_heads'): # 检查模块本身是否有 num_key_value_heads
            print(f"Warning: Using module.num_key_value_heads ({module.num_key_value_heads}) for num_heads in LlamaAttentionOTO.")
            self.num_heads = module.num_key_value_heads
        elif hasattr(module, 'config') and hasattr(module.config, 'num_key_value_heads'): # 检查配置中是否有
            print(f"Warning: Using module.config.num_key_value_heads ({module.config.num_key_value_heads}) for num_heads in LlamaAttentionOTO.")
            self.num_heads = module.config.num_key_value_heads
        else:
            # 收集更多诊断信息
            error_message = f"The LlamaAttention module (type: {type(module)}) " \
                            f"does not have 'num_heads', 'config.num_attention_heads', " \
                            f"or 'num_key_value_heads' (direct or via config). " \
                            f"Cannot determine the number of attention heads for OTO.\n"
            error_message += f"Available attributes in module: {dir(module)}\n"
            if hasattr(module, 'config'):
                error_message += f"Available attributes in module.config: {dir(module.config)}"
            raise AttributeError(error_message)
        
        # 仅当模块实例本身没有 hidden_size 时才添加
        if not hasattr(module, 'hidden_size'): 
            if hasattr(module, 'config') and hasattr(module.config, 'hidden_size'):
                module.hidden_size = module.config.hidden_size
                # print(f"LlamaAttentionOTO: INFO - Monkey-patched 'hidden_size' ({module.hidden_size}) onto LlamaAttention module instance.")
            else:
                raise AttributeError(f"LlamaAttentionOTO: Cannot get 'hidden_size' from module.config to patch onto LlamaAttention module instance.")
        
        super().__init__(id, _type, cfg_params, module, model) # 调用父类初始化

        # 确认 self.head_dim (LlamaAttention 模块有 head_dim 属性，父类应该能正确获取)
        if not hasattr(self, 'head_dim') or self.head_dim is None:
            if hasattr(self.module, 'head_dim'):
                self.head_dim = self.module.head_dim
            elif self.hidden_size is not None and self.num_heads is not None and self.num_heads > 0 : # 尝试从已知的 hidden_size 和 num_heads 推导
                self.head_dim = self.hidden_size // self.num_heads
                if self.hidden_size % self.num_heads != 0:
                     print(f"LlamaAttentionOTO: Warning - Derived head_dim ({self.head_dim}) is not an integer division of hidden_size by num_heads.")
            
            # 如果 head_dim 被重新计算/设置，并且剪枝模式是 'head_dim'，确保 num_groups 也更新
            if self.prune_mode == 'head_dim' and self.num_groups != self.head_dim:
                # print(f"LlamaAttentionOTO: INFO - Updating num_groups to match re-evaluated head_dim: {self.head_dim}")
                self.num_groups = self.head_dim

        self.out_key = 'o_proj'                         # Llama 中输出投影层的名称关键字
        self.op_name = 'llama_attention'                # 设置算子名称

    def prune_out_dim_head_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        """
        重写按头维度剪枝的方法，以适应 LlamaAttention 的特定逻辑。
        除了剪枝 Q, K, V 投影层，还需要重置旋转位置编码 (Rotary Embedding)。
        Args:
            pruned_idxes (list, optional): 需要在每个头内部剪枝的维度索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出投影层。默认为 True。
            **kwargs: 其他可选参数。
        """        
        # 情况一：如果用户提供了具体的参数名称列表 (param_names)，则只针对这些参数进行剪枝
        if len(param_names) > 0:
            visited_modules = set()         # 初始化一个集合，用于记录已经处理过的叶子模块，避免重复处理同一个模块多次
            # 遍历用户指定的需要剪枝的参数名称列表
            for param_name in param_names:
                # 遍历当前 LlamaAttentionOTO 算子内部的所有叶子模块 (例如 q_proj, k_proj, v_proj, o_proj 等对应的 OTO 算子)
                for module_name in self.leaf_modules:
                    # 检查当前的参数名称 (param_name) 是否属于当前遍历到的叶子模块 (module_name)
                    # 例如，检查 'model.layers.0.self_attn.q_proj.weight' 是否以 'model.layers.0.self_attn.q_proj' 开头
                    if not param_name.startswith(module_name):
                        # 如果参数名称不以叶子模块名称开头，说明该参数不属于此叶子模块，跳过，继续检查下一个叶子模块
                        continue
                    # 获取与该叶子模块名称对应的 OTO 算子对象 (例如 LinearOTO 或 LoraLinearOTO)
                    leaf_op = self.leaf_modules[module_name]
                    # 检查该叶子模块是否已经被处理过（是否已在 visited_modules 集合中）
                    if module_name not in visited_modules:
                        # 如果该叶子模块尚未处理过，则调用其 prune_out_dim 方法进行输出维度剪枝
                        # 注意：这里传递的 pruned_idxes 是相对于单个头的维度索引
                        # param_names=[param_name] 确保叶子算子的剪枝方法只处理当前指定的这一个参数
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    # 将当前处理过的叶子模块名称添加到 visited_modules 集合中，标记为已访问，防止后续重复处理
                    visited_modules.add(module_name)

        # 情况二：未提供参数名称列表，且跳过输出节点
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.module.head_dim)) - set(pruned_idxes))        # 计算保留的头内维度索引，使用 module 的 head_dim
            preserved_idxes.sort()
            self.module.head_dim = self.module.head_dim - len(pruned_idxes)                     # 更新模块的头维度属性
            self.module.reset_rotary_emb()                                                      # 重置旋转位置编码以匹配新的头维度
        
            # 遍历当前 LlamaAttentionOTO 算子内部的所有叶子模块 (例如 q_proj, k_proj, v_proj 等对应的 OTO 算子)
            for module_name in self.leaf_modules:
                # 检查当前叶子模块是否是输出投影层 (self.out_key 默认为 'o_proj')
                if self.out_key in module_name: 
                    # 如果是输出投影层，则跳过，不进行剪枝 (因为 skip_output_node 为 True)
                    continue
                # 获取与该叶子模块名称对应的 OTO 算子对象 (例如 LinearOTO 或 LoraLinearOTO)
                leaf_op = self.leaf_modules[module_name]
                # 初始化一个空列表，用于存储扩展后的全局剪枝索引，这些全局索引对应于 Q, K, V 投影层权重中需要被移除的行/列
                expand_pruned_idxes = list()
                # 遍历所有的头 (从 0 到 num_heads - 1)
                for h in range(self.num_heads):
                    # 对于当前的头 h，计算其在整个隐藏层维度中的全局剪枝索引，i是在单个头内部需要剪枝的维度索引 (来自 pruned_idxes)
                    # h * self.head_dim 是当前头 h 的起始偏移量
                    # 注意：这里的 self.head_dim 是指在本次剪枝操作 *之前* 的头维度，用于正确计算偏移量
                    # 将计算出的全局索引添加到 expand_pruned_idxes 列表中
                    expand_pruned_idxes.extend([i + h * self.head_dim for i in pruned_idxes]) 
                # 调用叶子算子 (leaf_op，例如 LinearOTO) 的 prune_out_dim 方法
                # 传入计算好的全局剪枝索引 expand_pruned_idxes，以剪枝 Q, K, 或 V 投影层的输出维度
                leaf_op.prune_out_dim(expand_pruned_idxes)
                
class BertAttentionOTO(BaseMultiHeadAttentionOTO):
    """
    BERT 模型注意力机制的算子类，继承自 BaseMultiHeadAttentionOTO。
    封装了 Hugging Face Transformers 库中的 BertAttention 模块，
    并提供了针对 BERT 结构的特定属性设置和剪枝逻辑。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None, model=None):
        """
        初始化 BertAttentionOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (BertAttention, optional): 关联的 BertAttention 模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params, module, model)                 # 调用父类 BaseMultiHeadAttentionOTO 的初始化方法
        self.out_key = 'output'                                         # 设置 BERT 中输出部分的名称关键字，用于识别输出相关的子模块（通常包含线性层和 LayerNorm）
        self.op_name = 'bert_attention'                                 # 设置算子名称为 'bert_attention'
        self.prune_mode = 'head_dim'                                    # BERT 注意力默认按头维度 (head_dim) 进行剪枝
        self.set_attributes()                                           # 调用设置属性的方法，初始化相关属性
        
    def set_attributes(self):
        """
        重写设置属性的方法，以从 BertAttention 模块的 'self' 子模块获取属性。
        BERT 的注意力参数和属性通常存储在名为 'self' 的子模块中。
        """
        self.num_heads = self.module.self.num_attention_heads           # 从关联模块的 'self' 子模块获取注意力头的数量
        self.head_dim = self.module.self.attention_head_size            # 从关联模块的 'self' 子模块获取每个注意力头的维度
        # 根据剪枝模式 (prune_mode) 设置 num_groups
        if self.prune_mode == 'head_dim':
            self.num_groups = self.head_dim                             # 如果按头维度剪枝，分组数等于头维度
        elif self.prune_mode == 'num_head':
            self.num_groups = self.num_heads                            # 如果按头数量剪枝，分组数等于头数量
        self.hidden_size = self.num_heads * self.head_dim               # 计算隐藏层大小 (等于 头数 * 头维度)
        self.num_group_divisible = 2                                    # 设置分组可除数，用于确保剪枝后的维度是有效的（例如，某些操作要求维度是偶数）

    def prune_out_dim_head_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        """
        重写按头维度 (head_dim) 剪枝的方法，以适应 BertAttention 的特定逻辑。
        除了调用叶子模块的剪枝方法外，还需要更新 'self' 子模块的 
        attention_head_size 和 all_head_size 属性。
        Args:
            pruned_idxes (list, optional): 需要在每个头内部剪枝的维度索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出部分 ('output' 子模块)。默认为 True。
            **kwargs: 其他可选参数。
        """        
        # 情况一：如果用户提供了具体的参数名称列表 (param_names)，则只针对这些参数进行剪枝
        if len(param_names) > 0:
            visited_modules = set()     # 初始化一个集合，用于记录已经处理过的叶子模块，避免重复处理
            # 遍历用户指定的参数名称
            for param_name in param_names:
                # 遍历当前算子内部的所有叶子模块 (如 query, key, value 对应的 OTO 算子)
                for module_name in self.leaf_modules:
                    # 检查参数名称是否属于当前叶子模块
                    if not param_name.startswith(module_name):
                        continue # 不属于则跳过
                    leaf_op = self.leaf_modules[module_name]    # 获取叶子模块对应的 OTO 算子对象
                    # 如果该叶子模块尚未处理过
                    if module_name not in visited_modules:
                        # 调用叶子算子的 prune_out_dim 方法，传入头内剪枝索引和当前参数名
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    visited_modules.add(module_name)        # 标记该叶子模块为已处理
        # 情况二：未提供参数名称列表，且跳过输出节点 (skip_output_node is True)
        elif len(param_names) == 0 and skip_output_node:
            # 计算需要保留的头内维度索引
            # self.num_groups 在 head_dim 模式下等于 self.head_dim
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes)) 
            preserved_idxes.sort() # 对索引排序
            # 更新 BertAttention 模块 'self' 子模块的头维度属性
            self.module.self.attention_head_size = self.head_dim - len(pruned_idxes)
            # 更新 BertAttention 模块 'self' 子模块的总头大小属性
            self.module.self.all_head_size = self.module.self.num_attention_heads * self.module.self.attention_head_size
            
            # 遍历叶子模块，对 Q, K, V 投影层进行剪枝
            for module_name in self.leaf_modules:
                # 如果叶子模块属于输出部分 (由 self.out_key 定义)，则跳过
                if self.out_key in module_name: 
                    continue
                # 获取叶子模块对应的 OTO 算子对象
                leaf_op = self.leaf_modules[module_name]
                # 初始化列表，用于存储扩展后的全局剪枝索引
                expand_pruned_idxes = list()
                # 遍历所有头
                for h in range(self.num_heads):
                    # 将头内剪枝索引 (pruned_idxes) 映射到全局索引
                    # i 是头内索引，h * self.head_dim 是当前头的偏移量 (使用剪枝前的 head_dim)
                    expand_pruned_idxes.extend([i + h * self.head_dim for i in pruned_idxes]) 
                # 调用叶子算子的 prune_out_dim 方法，传入扩展后的全局索引
                leaf_op.prune_out_dim(expand_pruned_idxes)

    def prune_out_dim_num_head(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        """
        重写按头数量 (num_head) 剪枝的方法，以适应 BertAttention 的特定逻辑。
        除了调用叶子模块的剪枝方法外，还需要更新 'self' 子模块的 
        num_attention_heads 和 all_head_size 属性。
        Args:
            pruned_idxes (list, optional): 需要剪枝的头的索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出部分 ('output' 子模块)。默认为 True。
            **kwargs: 其他可选参数。
        """
        # 初始化一个集合，用于记录已经处理过的叶子模块
        visited_modules = set() 
        # 情况一：如果用户提供了具体的参数名称列表 (param_names)
        if len(param_names) > 0:
            # 遍历用户指定的参数名称
            for param_name in param_names:
                # 遍历当前算子内部的所有叶子模块
                for module_name in self.leaf_modules:
                    # 检查参数名称是否属于当前叶子模块
                    if not param_name.startswith(module_name):
                        continue # 不属于则跳过
                    # 获取叶子模块对应的 OTO 算子对象
                    leaf_op = self.leaf_modules[module_name]
                    # 如果该叶子模块尚未处理过
                    if module_name not in visited_modules:
                        # 调用叶子算子的 prune_out_dim 方法，传入头的剪枝索引和当前参数名
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    # 标记该叶子模块为已处理
                    visited_modules.add(module_name)
        # 情况二：未提供参数名称列表，且跳过输出节点 (skip_output_node is True)
        elif len(param_names) == 0 and skip_output_node:
            # 计算需要保留的头索引
            # self.num_groups 在 num_head 模式下等于 self.num_heads
            preserved_idxes = list(set(range(self.num_groups)) - set(pruned_idxes)) 
            preserved_idxes.sort() # 对索引排序
            # 更新 BertAttention 模块 'self' 子模块的头数量属性
            self.module.self.num_attention_heads = self.num_groups - len(pruned_idxes)
            # 更新 BertAttention 模块 'self' 子模块的总头大小属性
            self.module.self.all_head_size = self.module.self.num_attention_heads * self.module.self.attention_head_size
            
            # 遍历叶子模块，对 Q, K, V 投影层进行剪枝
            for module_name in self.leaf_modules:
                # 如果叶子模块属于输出部分 (由 self.out_key 定义)，则跳过
                if self.out_key in module_name: 
                    continue
                # 获取叶子模块对应的 OTO 算子对象
                leaf_op = self.leaf_modules[module_name]
                # 初始化列表，用于存储扩展后的全局剪枝索引
                expand_pruned_idxes = list()
                # 确保 pruned_idxes 是列表或可以转换为列表，以便迭代
                # 如果 pruned_idxes 是 numpy array，调用 tolist()
                # 遍历需要剪枝的头的索引 i
                for i in pruned_idxes.tolist():
                    # 对于每个要剪枝的头 i，将其对应的所有维度索引添加到 expand_pruned_idxes
                    for h in range(self.head_dim):
                        # h 是头内的维度索引，i * self.head_dim 是头 i 的起始偏移量
                        expand_pruned_idxes.append(h + i * self.head_dim)
                # 调用叶子算子的 prune_out_dim 方法，传入扩展后的全局索引
                leaf_op.prune_out_dim(expand_pruned_idxes)

class PhiAttentionOTO(BaseMultiHeadAttentionOTO):
    """
    Phi 模型注意力机制的算子类，继承自 BaseMultiHeadAttentionOTO。
    Phi 模型可能使用了特定的 MHA 实现 (例如 PhiMHA)。
    提供了针对 Phi 结构的特定属性设置和剪枝逻辑。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None):
        """
        初始化 PhiAttentionOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (PhiMHA, optional): 关联的 PhiMHA 模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params, module)     # 调用父类 BaseMultiHeadAttentionOTO 的初始化方法
        self.is_stem = True                                 # 标记为 Stem 算子，表示可以转换主维度
        self.is_basic = False                               # 标记为非基础模块（因为是复合模块）
        self.out_key = 'out_proj'                           # 设置 Phi 中输出投影层的名称关键字
        self.op_name = 'phi_attention'                      # 设置算子名称为 'phi_attention'
        self.set_attributes()                               # 调用 set_attributes 方法设置 Phi 特定的属性

    def set_attributes(self):
        """
        设置 Phi 注意力相关的属性。
        从关联的 PhiMHA 模块获取头数和头维度。
        """
        self.num_heads = self.module.n_head     # 从关联模块获取头数 (属性名为 n_head)
        self.head_dim = self.module.head_dim    # 从关联模块获取头维度 (属性名为 head_dim)
        # 根据剪枝模式 (prune_mode) 设置 num_groups
        if self.prune_mode == 'head_dim':
            self.num_groups = self.head_dim     # 如果按头维度剪枝，分组数等于头维度
        elif self.prune_mode == 'num_head':
            self.num_groups = self.num_heads    # 如果按头数量剪枝，分组数等于头数量

    def get_param_groups(self, param_names=list(), skip_output_node=False, **kwargs):
        """
        获取 Phi 注意力算子的参数分组信息。
        与基类类似，但此实现似乎只收集了输出投影层的参数组信息。
        可能需要根据 PhiMHA 的具体结构进行调整以包含 Q, K, V 等参数。
        Args:
            param_names (list, optional): 要包含的参数名称列表。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出投影层。默认为 False。
            **kwargs: 其他可选参数。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                           # 初始化参数组字典
        param_groups['op'] = self.op_name               # 设置操作类型
        param_groups['p_names'] = list()                # 初始化参数名称列表
        param_groups['params'] = list()                 # 初始化参数对象列表
        param_groups['p_transform'] = list()            # 初始化参数变换类型列表
        param_groups['num_heads'] = self.num_heads      # 添加头数信息
        param_groups['head_dim'] = self.head_dim        # 添加头维度信息
        # 如果存在 LoRA 缩放因子，添加到参数组信息中
        if hasattr(self, 'lora_scaling'):
            param_groups['lora_scaling'] = self.lora_scaling
        # 确定目标参数名称：如果提供了 param_names 则使用它，否则使用算子所有的参数名称
        target_param_names = param_names if len(param_names) > 0 else self.name_to_param.keys()
        # 遍历目标参数名称
        for p_name in target_param_names:
            param = self.name_to_param[p_name]# 获取参数对象
            # 只处理输出投影层相关的参数 (如果未被跳过)
            if self.out_key in p_name and not skip_output_node:
                param_groups['p_names'].append(p_name)      # 添加参数名称
                param_groups['params'].append(param)        # 添加参数对象
                # 如果是 LoRA A 矩阵，设置为不剪枝
                if 'lora_A' in p_name:
                    param_groups['p_transform'].append(TensorTransform.NO_PRUNE)
                # 否则，设置为基础变换类型
                else:
                    param_groups['p_transform'].append(TensorTransform.BASIC)
            # 注意：此实现似乎只收集了输出投影层的参数组信息。
            # 如果需要包含Q, K, V等投影层的参数，需要修改此处的逻辑，
            # 类似于 BaseMultiHeadAttentionOTO.get_param_groups 中的处理方式。
        return param_groups     # 返回参数组字典

    def prune_out_dim_head_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):  
        """
        重写按头维度 (head_dim) 剪枝的方法，以适应 PhiAttention 的特定逻辑。
        更新模块的 head_dim 属性，并调用叶子模块的剪枝方法。
        Args:
            pruned_idxes (list, optional): 需要在每个头内部剪枝的维度索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出投影层。默认为 True。
            **kwargs: 其他可选参数。
        """  
        visited_modules = set()     # 初始化一个集合，用于记录已经处理过的叶子模块
        # 情况一：如果用户提供了具体的参数名称列表 (param_names)
        if len(param_names) > 0:
            # 遍历用户指定的参数名称
            for param_name in param_names:
                # 遍历当前算子内部的所有叶子模块
                for module_name in self.leaf_modules:
                    # 检查参数名称是否属于当前叶子模块
                    if not param_name.startswith(module_name):
                        continue # 不属于则跳过
                    # 获取叶子模块对应的 OTO 算子对象
                    leaf_op = self.leaf_modules[module_name]
                    # 如果该叶子模块尚未处理过
                    if module_name not in visited_modules:
                        # 调用叶子算子的 prune_out_dim 方法，传入头内剪枝索引和当前参数名
                        leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                    # 标记该叶子模块为已处理
                    visited_modules.add(module_name)
        # 情况二：未提供参数名称列表，且跳过输出节点 (skip_output_node is True)
        elif len(param_names) == 0 and skip_output_node:
            # 计算需要保留的头内维度索引，使用关联模块的 head_dim
            preserved_idxes = list(set(range(self.module.head_dim)) - set(pruned_idxes)) 
            preserved_idxes.sort() # 对索引排序
            # 更新关联模块的头维度属性
            self.module.head_dim = self.module.head_dim - len(pruned_idxes)
            # 遍历叶子模块，对 Q, K, V 投影层（或其他相关层）进行剪枝
            for module_name in self.leaf_modules:
                # 如果叶子模块是输出投影层 (由 self.out_key 定义)，则跳过
                if self.out_key in module_name:
                    continue
                # 获取叶子模块对应的 OTO 算子对象
                leaf_op = self.leaf_modules[module_name]
                # 初始化列表，用于存储扩展后的全局剪枝索引
                expand_pruned_idxes = list()
                # 遍历所有头
                for h in range(self.num_heads):
                    # 将头内剪枝索引 (pruned_idxes) 映射到全局索引
                    # i 是头内索引，h * self.head_dim 是当前头的偏移量 (使用剪枝前的 head_dim)
                    expand_pruned_idxes.extend([i + h * self.head_dim for i in pruned_idxes])
                # 调用叶子算子的 prune_out_dim 方法，传入扩展后的全局索引
                leaf_op.prune_out_dim(expand_pruned_idxes)

class VitAttentionOTO(BaseMultiHeadAttentionOTO):
    """
    ViT (Vision Transformer) 模型注意力机制的算子类，继承自 BaseMultiHeadAttentionOTO。
    ViT 使用单个 qkv 线性层同时生成 Q、K、V，与传统的分离式 Q、K、V 投影不同。
    提供了针对 ViT 结构的特定属性设置和剪枝逻辑。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None, model=None):
        """
        初始化 VitAttentionOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (Attention, optional): 关联的 ViT Attention 模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params, module, model)     # 调用父类 BaseMultiHeadAttentionOTO 的初始化方法
        self.is_stem = True                                 # 标记为 Stem 算子，表示可以转换主维度
        self.is_basic = False                               # 标记为非基础模块（因为是复合模块）
        self.out_key = 'proj'                               # 设置 ViT 中输出投影层的名称关键字
        self.op_name = 'vit_attention'                      # 设置算子名称为 'vit_attention'
        self.prune_mode = 'head_dim'                        # 默认按头维度剪枝
        self.set_attributes()                               # 调用 set_attributes 方法设置 ViT 特定的属性

    def set_attributes(self):
        """
        设置 ViT 注意力相关的属性。
        从关联的 ViT Attention 模块获取头数和头维度。
        """
        # 尝试从不同的属性名获取头数
        for name, module in self.root_model.named_modules():
            if 'attn' in name and hasattr(module, 'num_heads'):
                # print(f"Found Attention module: {name}")
                
                # 验证头数获取
                if hasattr(module, 'num_heads'):
                    self.num_heads = module.num_heads
                    # print(f"  num_heads (direct): {self.num_heads}")
                elif hasattr(module, 'n_heads'):
                    self.num_heads = module.n_heads
                    # print(f"  num_heads (n_heads): {self.num_heads}")
                elif hasattr(module, 'num_attention_heads'):
                    self.num_heads = module.num_attention_heads
                    # print(f"  num_heads (num_attention_heads): {self.num_heads}")
                
                # 验证头维度获取
                if hasattr(module, 'head_dim'):
                    self.head_dim = module.head_dim
                    # print(f"  head_dim (direct): {self.head_dim}")
                elif hasattr(module, 'attention_head_size'):
                    self.head_dim = module.attention_head_size
                    # print(f"  head_dim (attention_head_size): {self.head_dim}")
                else:
                    # 从qkv层计算
                    qkv_out_features = module.qkv.out_features
                    embed_dim = qkv_out_features // 3
                    self.head_dim = embed_dim // self.num_heads
                    # print(f"  head_dim (calculated): {self.head_dim}")
                
                # 验证隐藏维度
                if hasattr(module, 'embed_dim'):
                    self.hidden_size = module.embed_dim
                    # print(f"  embed_dim (direct): {self.hidden_size}")
                else:
                    qkv_out_features = module.qkv.out_features
                    self.hidden_size = qkv_out_features // 3
                    # print(f"  embed_dim (from qkv): {self.hidden_size}")
                
                # print(f"  Verification: head_dim * num_heads = {self.head_dim * self.num_heads} vs embed_dim = {self.hidden_size}")
    
        # 根据剪枝模式设置 num_groups
        if self.prune_mode == 'head_dim':
            self.num_groups = self.head_dim     # 如果按头维度剪枝，分组数等于头维度
        elif self.prune_mode == 'num_head':
            self.num_groups = self.num_heads    # 如果按头数量剪枝，分组数等于头数量

        self.num_group_divisible = 2                                    # 设置分组可除数

    def prune_out_dim_head_dim(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        """
        按头维度(head_dim)剪枝ViT注意力算子的输出维度。
        考虑ViT中qkv联合投影的特殊性，需要特别处理。
        Args:
            pruned_idxes (list, optional): 需要在每个头内部剪枝的维度索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出投影层。默认为True。
            **kwargs: 其他可选参数。
        """
        visited_modules = set()  # 记录已处理的模块
        
        # 情况一：指定了具体参数名
        if len(param_names) > 0:
            for param_name in param_names:
                for module_name in self.leaf_modules:
                    if not param_name.startswith(module_name):
                        continue
                        
                    leaf_op = self.leaf_modules[module_name]
                    if module_name not in visited_modules:
                        # 处理qkv层的特殊情况
                        if 'qkv' in module_name:
                            # 需要扩展pruned_idxes以适应qkv三合一的特性
                            qkv_pruned_idxes = []
                            for idx in pruned_idxes:
                                # 为Q、K、V部分分别添加对应索引
                                # Q部分
                                qkv_pruned_idxes.extend([idx + h * self.head_dim for h in range(self.num_heads)])
                                # K部分
                                qkv_pruned_idxes.extend([idx + h * self.head_dim + self.hidden_size for h in range(self.num_heads)])
                                # V部分
                                qkv_pruned_idxes.extend([idx + h * self.head_dim + 2 * self.hidden_size for h in range(self.num_heads)])
                            
                            leaf_op.prune_out_dim(qkv_pruned_idxes, param_names=[param_name])
                        else:
                            # 其他层正常处理
                            leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                            
                    visited_modules.add(module_name)
                    
        # 情况二：未指定参数名且跳过输出节点
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.head_dim)) - set(pruned_idxes))
            preserved_idxes.sort()
            
            # 更新模块的head_dim属性
            new_head_dim = self.head_dim - len(pruned_idxes)
            if hasattr(self.module, 'head_dim'):
                self.module.head_dim = new_head_dim
            
            # 处理所有叶子模块
            for module_name in self.leaf_modules:
                # 跳过输出投影层
                if self.out_key in module_name:
                    continue
                    
                leaf_op = self.leaf_modules[module_name]
                
                # 特别处理qkv层
                if 'qkv' in module_name:
                    # 需要同时剪枝Q、K、V三部分
                    qkv_pruned_idxes = []
                    for h in range(self.num_heads):
                        # Q部分的索引
                        q_offset = h * self.head_dim
                        qkv_pruned_idxes.extend([i + q_offset for i in pruned_idxes])
                        
                        # K部分的索引
                        k_offset = self.hidden_size + h * self.head_dim
                        qkv_pruned_idxes.extend([i + k_offset for i in pruned_idxes])
                        
                        # V部分的索引
                        v_offset = 2 * self.hidden_size + h * self.head_dim
                        qkv_pruned_idxes.extend([i + v_offset for i in pruned_idxes])
                    
                    # 剪枝qkv层
                    leaf_op.prune_out_dim(qkv_pruned_idxes)
                else:
                    # 其他层使用标准的扩展索引方法
                    expand_pruned_idxes = []
                    for h in range(self.num_heads):
                        expand_pruned_idxes.extend([i + h * self.head_dim for i in pruned_idxes])
                    leaf_op.prune_out_dim(expand_pruned_idxes)
                    
            # 更新属性
            self.head_dim = new_head_dim
            if self.prune_mode == 'head_dim':
                self.num_groups = self.head_dim

    def prune_out_dim_num_head(self, pruned_idxes=list(), param_names=list(), skip_output_node=True, **kwargs):
        """
        按头数量(num_head)剪枝ViT注意力算子的输出维度。
        考虑ViT中qkv联合投影的特殊性，需要特别处理。
        Args:
            pruned_idxes (list, optional): 需要剪枝的头索引列表。默认为空列表。
            param_names (list, optional): 如果提供，则只剪枝指定的参数。默认为空列表。
            skip_output_node (bool, optional): 是否跳过输出投影层。默认为True。
            **kwargs: 其他可选参数。
        """
        visited_modules = set()  # 记录已处理的模块
        
        # 情况一：指定了具体参数名
        if len(param_names) > 0:
            for param_name in param_names:
                for module_name in self.leaf_modules:
                    if not param_name.startswith(module_name):
                        continue
                        
                    leaf_op = self.leaf_modules[module_name]
                    if module_name not in visited_modules:
                        # 处理qkv层的特殊情况
                        if 'qkv' in module_name:
                            qkv_pruned_idxes = []
                            for i in pruned_idxes:
                                # Q部分：剪枝整个头对应的所有维度
                                q_start = i * self.head_dim
                                q_end = q_start + self.head_dim
                                qkv_pruned_idxes.extend(list(range(q_start, q_end)))
                                
                                # K部分：剪枝整个头对应的所有维度
                                k_start = self.hidden_size + i * self.head_dim
                                k_end = k_start + self.head_dim
                                qkv_pruned_idxes.extend(list(range(k_start, k_end)))
                                
                                # V部分：剪枝整个头对应的所有维度
                                v_start = 2 * self.hidden_size + i * self.head_dim
                                v_end = v_start + self.head_dim
                                qkv_pruned_idxes.extend(list(range(v_start, v_end)))
                                
                            leaf_op.prune_out_dim(qkv_pruned_idxes, param_names=[param_name])
                        else:
                            # 其他层正常处理
                            leaf_op.prune_out_dim(pruned_idxes, param_names=[param_name])
                            
                    visited_modules.add(module_name)
                    
        # 情况二：未指定参数名且跳过输出节点
        elif len(param_names) == 0 and skip_output_node:
            preserved_idxes = list(set(range(self.num_heads)) - set(pruned_idxes))
            preserved_idxes.sort()
            
            # 更新模块的num_heads属性
            new_num_heads = self.num_heads - len(pruned_idxes)
            if hasattr(self.module, 'num_heads'):
                self.module.num_heads = new_num_heads
            
            # 处理所有叶子模块
            for module_name in self.leaf_modules:
                # 跳过输出投影层
                if self.out_key in module_name:
                    continue
                    
                leaf_op = self.leaf_modules[module_name]
                
                # 特别处理qkv层
                if 'qkv' in module_name:
                    qkv_pruned_idxes = []
                    for i in pruned_idxes:
                        # 对于每个被剪枝的头，我们需要剪枝其在Q、K、V三部分的所有维度
                        
                        # Q部分：剪枝整个头的所有维度
                        q_start = i * self.head_dim
                        q_end = q_start + self.head_dim
                        qkv_pruned_idxes.extend(list(range(q_start, q_end)))
                        
                        # K部分：剪枝整个头的所有维度
                        k_start = self.hidden_size + i * self.head_dim
                        k_end = k_start + self.head_dim
                        qkv_pruned_idxes.extend(list(range(k_start, k_end)))
                        
                        # V部分：剪枝整个头的所有维度
                        v_start = 2 * self.hidden_size + i * self.head_dim
                        v_end = v_start + self.head_dim
                        qkv_pruned_idxes.extend(list(range(v_start, v_end)))
                    
                    # 剪枝qkv层
                    leaf_op.prune_out_dim(qkv_pruned_idxes)
                else:
                    # 其他层使用标准的扩展索引方法
                    expand_pruned_idxes = []
                    for i in pruned_idxes:
                        # 对于每个被剪枝的头，将其所有维度都添加到剪枝列表
                        head_start = i * self.head_dim
                        head_end = head_start + self.head_dim
                        expand_pruned_idxes.extend(list(range(head_start, head_end)))
                    leaf_op.prune_out_dim(expand_pruned_idxes)

            # 更新属性
            self.num_heads = new_num_heads
            if self.prune_mode == 'num_head':
                self.num_groups = self.num_heads
    

class PReLUOTO(Operator):
    """
    PReLU 激活函数的算子类，封装了 PyTorch 的 nn.PReLU 模块。
    提供了参数管理、参数剪枝和参数分组功能。
    PReLU 的可学习参数通常被视为辅助参数，其剪枝与关联的主层（如卷积层）同步。
    """
    def __init__(self, id=None, _type=None, cfg_params=dict(), module=None): # Corrected method name
        """
        初始化 PReLUOTO 算子。
        Args:
            id (str, optional): 算子的唯一标识符。默认为 None。
            _type (str, optional): 算子的类型名称。默认为 None。
            cfg_params (dict, optional): 算子的配置参数。默认为空字典。
            module (nn.PReLU, optional): 关联的 PyTorch PReLU 模块。默认为 None。
        """
        super().__init__(id, _type, cfg_params, module)     # 调用父类 Operator 的初始化方法
        self.is_stem = False                                # 标记为非 Stem 算子，PReLU 不改变特征维度
        self.set_num_groups()                               # 设置分组数，对于 PReLU，通常是可学习参数的数量 (num_parameters)
        self.p_transform = TensorTransform.ACCESSORY        # 设置参数变换类型为 ACCESSORY，表示其剪枝依赖于其他主层
    def get_param_groups(self, param_names=[]):
        """
        获取 PReLU 算子的参数分组信息。
        Args:
            param_names (list, optional): 要包含的参数名称列表。默认为空列表。
        Returns:
            dict: 包含参数分组信息的字典。
        """
        param_groups = dict()                       # 初始化参数组字典
        param_groups['op'] = 'prelu'                # 设置操作类型为 'prelu'
        param_groups['p_names'] = list()            # 初始化参数名称列表
        param_groups['params'] = list()             # 初始化参数对象列表
        param_groups['p_transform'] = list()        # 初始化参数变换类型列表
        # 遍历指定的参数名称 (通常只有一个 'weight' 参数)
        for p_name in param_names:
            # 检查参数是否存在于算子的参数映射中
            if p_name in self.name_to_param:
                param_groups['p_names'].append(p_name)                      # 添加参数名称
                param_groups['params'].append(self.name_to_param[p_name])   # 添加参数对象
                param_groups['p_transform'].append(self.p_transform)        # 添加参数变换类型 (ACCESSORY)
        return param_groups                                                 # 返回参数组字典

    def prune_out_dim(self, pruned_idxes=list(), **kwargs):
        """
        剪枝 PReLU 的参数维度。
        这会影响 num_parameters 属性和 weight 参数。
        剪枝操作通常与前一个卷积层或线性层的输出通道/特征剪枝同步。
        Args:
            pruned_idxes (list, optional): 需要剪枝的参数索引列表。默认为空列表。
            **kwargs: 额外的关键字参数。
        """
        # 计算需要保留的参数索引，self.module.num_parameters 通常等于输入的通道数或特征数
        preserved_idxes = list(set(range(self.module.num_parameters)) - set(pruned_idxes))
        preserved_idxes.sort() # 对索引排序
        # 剪枝 PReLU 的 weight 参数 (dim=0) 及其梯度
        self.module.weight = self.prune_param_and_grad(self.module.weight, preserved_idxes, 0)
        # 更新模块的 num_parameters 属性以反映剪枝后的数量
        self.module.num_parameters = self.module.num_parameters - len(pruned_idxes)   



# ================== 模块映射字典 ==================

# 基础模块：可以直接映射到 OTO 算子类的 PyTorch 模块
# Key 是 PyTorch 模块的类名字符串，Value 是对应的 OTO 算子类
BASIC_MODULES = {
    # 基本算子类型与其对应的 OTO 类的映射
    'ConvTranspose2d': ConvTranspose2dOTO,      # 反卷积 (转置卷积)
    'Conv2d': Conv2dOTO,                        # 2D 卷积
    'ModulatedConv2d': Conv2dOTO,               # 特殊卷积（例如 StyleGAN 中的 ModulatedConv2d，视为 Conv2d 处理）
    'EqualLinear': LinearOTO,                   # 特殊线性层（例如 StyleGAN 中的 EqualLinear，视为 Linear 处理）
    'Linear': LinearOTO,                        # 标准线性层 (全连接层)
    'BatchNorm2d': BatchNormOTO,                # 2D 批归一化
    'InstanceNorm2d': InstanceNormOTO,          # 2D 实例归一化
    'GroupNorm': GroupNormOTO,                  # 分组归一化
    'Embedding': EmbeddingOTO,                  # 嵌入层
    'LlamaRMSNorm': LayerNormOTO,               # Llama RMSNorm (视为 LayerNorm 处理)
    'LayerNorm': LayerNormOTO,                  # 标准层归一化
    'PReLU': PReLUOTO,                          # PReLU 激活函数 (带可学习参数)
}

# 复合模块：包含多个子模块或需要特殊处理的模块，映射到特定的复合 OTO 算子类
# 通常要求包含至少两个可训练变量的节点，以区别于简单的包装器
# Key 是 PyTorch 模块的类名字符串，Value 是对应的 OTO 算子类
COMPOSED_MODULES = {
    # 复合算子类型与其对应的 OTO 类的映射
    'LlamaAttention': LlamaAttentionOTO,                # Llama 注意力机制模块
    'SelfAttention': BaseMultiHeadAttentionOTO,         # 通用多头自注意力 (作为基类或默认实现)
    'BertAttention': BertAttentionOTO,                  # Bert 注意力机制模块
    'PhiMHA': PhiAttentionOTO,                          # Phi 多头注意力机制模块
    'LoraLinear': LoraLinearOTO,                        # 应用了 LoRA 的线性层
    'LoraEmbedding': LoraEmbeddingOTO,                  # 应用了 LoRA 的嵌入层
    'Attention': VitAttentionOTO,                       # ViT 注意力机制模块 (timm 中的名称)
}

# 不支持剪枝或暂不支持的基础算子类型名称列表，如果某个计算图节点组包含这些类型的算子，该组可能被标记为不可剪枝
UNPRUNABLE_BASIC_OPERATORS = [
    'depthtospace',     # 空间到深度操作 (通常是函数式 API 调用，没有独立的 nn.Module)
]

# 不支持剪枝或暂不支持的复合算子类型名称列表，包含这些算子的节点组可能被标记为不可剪枝，特别是 LoRA 相关层，其结构通常不参与剪枝
UNPRUNABLE_COMPOSED_OPERATORS = [
    'LoraLinearOTO',    # LoRA 线性层 (LoRA 参数通常固定或有特定优化策略，不参与结构剪枝)
    'LoraEmbeddingOTO'  # LoRA 嵌入层 (同上)
]

