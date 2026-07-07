import numpy as np

class Node:
    """
    表示计算图中的一个节点。
    每个节点代表一个操作（op），并包含有关其输入、输出、参数和形状的信息。
    """
    def __init__(self, id=None, op_name="", op=None, inputs=[], outputs=[], param_names=[], output_shape=[]):
        """
        初始化 Node 对象。
        Args:
            id: 节点的唯一标识符。
            op_name (str): 操作的名称（例如，"Conv", "Linear"）。
            op: 与此节点关联的操作对象（可能包含配置参数）。
            inputs (list): 输入节点的 ID 列表。
            outputs (list): 输出节点的 ID 列表。
            param_names (list): 与此节点操作相关的参数名称列表。
            output_shape (list): 节点输出的形状。
        """
        super().__init__()          # 没有显式地指定任何父类。object.__init__方法本身通常不做任何特别的事情（它是一个空操作），但调用super().__init__()是一种良好的编程实践
        self.id = id                # 节点 ID
        self.op = op                # 关联的操作对象
        self.op_name = op_name      # 操作名称
        self.inputs = ['node-' + str(i) for i in inputs]        # 将输入 ID 格式化为 'node-<id>'
        self.outputs = ['node-' + str(o) for o in outputs]      # 将输出 ID 格式化为 'node-<id>'
        self.param_names = param_names                          # 参数名称列表
        self.node_group_ids = list()                            # 用于节点分组的 ID 列表（例如，用于剪枝）
        # 跟踪节点的剪枝状态（输入/输出维度是否已被剪枝）
        self.pruned_status = {
            "out_dim": False,
            "in_dim": False
        }
        self.output_shape = output_shape        # 节点输出张量的形状
        self.input_shape = []                   # 节点输入张量的形状（可能在稍后填充）

    def __repr__(self) -> str:
        """
        返回节点的字符串表示形式。
        Returns:
            str: 包含节点 ID、操作名称和参数名称的字符串。
        """
        return f"Node id: {self.id}, op_name: {self.op_name}, param_names: {self.param_names}"

    @property
    def title(self):
        """
        title 被定义为一个方法，但使用 @property 装饰器后，它可以像一个属性一样被访问，用来获取节点的描述性标题。
        生成节点的描述性标题。
        标题通常包含操作名称，对于卷积等操作，还会包含核大小和步长等信息。
        Returns:
            str: 节点的描述性标题。
        """
        # 如果没有关联的操作对象，则直接返回操作名称
        if not self.op:
            return self.op_name
        # 默认标题：如果 op_name 和 op._type 不同，则为 "op_name-op._type"，否则为 "op._type"
        title = (self.op_name + '-' + self.op._type) if self.op_name != self.op._type else self.op._type
        # 如果操作配置参数中包含 "kernel_shape"
        if "kernel_shape" in self.op.cfg_params:
            kernel = self.op.cfg_params["kernel_shape"]     # 获取核大小
            title += "x".join(map(str, kernel))             # 将核大小添加到标题，用 "x" 连接（例如，"3x3"）
        # 如果操作配置参数中包含 "stride"
        if "stride" in self.op.cfg_params:
            stride = self.op.cfg_params["stride"]       # 获取步长
            # 如果步长在所有维度上都相同，则简化为单个值
            if np.unique(stride).size == 1:
                stride = stride[0]
            # 如果步长不为 1，则将其添加到标题（例如，"/s2"）
            if stride != 1:
                title += "/s{}".format(str(stride))
        return title

    def is_stem(self):
        """
        检查节点是否是模型的 "stem" 部分（通常是初始的卷积或线性层）。
        Returns:
            bool: 如果节点是 stem 则为 True，否则为 False。
        """
        if self.op is not None:
            # 如果节点有操作
            if self.op.is_basic:
                # 如果操作是基本操作，则检查其 is_stem 属性
                return self.op.is_stem
            # 否则，检查它是否是卷积、转置卷积或线性层，是返回true，否则返回false
            else:
                return self.is_conv() or self.is_convtranspose() or self.is_linear()
        else:
            return False        # 如果没有操作对象，则不是 stem

    def is_conv(self):
        """
        检查节点是否是卷积操作。
        Returns:
            bool: 如果节点是卷积操作则为 True，否则为 False。
        """
        return self.op_name == "Conv" or self.op_name == 'conv'

    def is_convtranspose(self):
        """
        检查节点是否是转置卷积操作。
        Returns:
            bool: 如果节点是转置卷积操作则为 True，否则为 False。
        """
        return self.op_name == "ConvTranspose" or self.op_name == 'convtranspose'

    def is_linear(self):
        """
        检查节点是否是线性（全连接）或 Gemm 操作。
        Returns:
            bool: 如果节点是线性或 Gemm 操作则为 True，否则为 False。
        """
        return self.op_name == "Linear" or self.op_name == 'linear' \
            or self.op_name == "Gemm" or self.op_name == "gemm"

    def is_concat(self, axis=None):
        """
        检查节点是否是连接（Concat）操作。
        Args:
            axis (int, optional): 如果提供，则检查连接操作是否沿指定轴进行。默认为 None。
        Returns:
            bool: 如果节点是连接操作（并且如果指定了 axis，则轴匹配），则为 True，否则为 False。
        """ 
		# TODO Need to double check if self.op_name == cat, 
        # then onnx trace is not enabled, skip check axix constraint
        if self.op_name == 'cat':
            return True
		
        _is_concat = self.op_name == "Concat" or self.op_name == 'concat'       # 首先检查操作名称是否为 "Concat" 或 "concat"
        # 如果未指定 axis，则仅返回是否为连接操作
        if axis == None:
            return _is_concat
        # 如果指定了 axis，并且是连接操作，并且操作对象具有 cfg_params 属性
        if _is_concat and hasattr(self.op, 'cfg_params'):
            # 检查 cfg_params 中是否有 'axis' 键
            if 'axis' in self.op.cfg_params:
                return True if self.op.cfg_params['axis'] == axis else False    # 返回操作的 axis 是否与指定的 axis 匹配
            else:
                # 如果没有 'axis' 键，则无法判断轴是否匹配
                return False
        return _is_concat       # 如果不是连接操作，或者不满足上述条件，则返回 False

    def is_dummy(self):
        """
        检查节点是否是虚拟输入或输出节点。
        Returns:
            bool: 如果节点 ID 是 'dummy_input' 或 'dummy_output' 则为 True，否则为 False。
        """
        return True if self.id == 'dummy_input' or self.id == 'dummy_output' else False


