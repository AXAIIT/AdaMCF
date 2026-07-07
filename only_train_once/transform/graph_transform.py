import imp
import re
from . import ge
from only_train_once.graph.node  import Node

class Rename():
    """
    节点重命名变换类。
    用于批量修改计算图中节点的操作名（op_name），如将ONNX导出的"onnx::Conv"重命名为"Conv"。
    op：代表节点的操作类型，比如 "Conv"、"BatchNorm"、"Relu" 等。通常对应于节点的 op_name 属性，即节点执行的算子类型。用于批量匹配和替换所有属于某种操作类型的节点。
    name：代表节点的名字，比如 "conv1"、"bn1"、"layer3.0.conv2" 等。通常对应于节点的 name 或 id 属性，是网络结构中每个节点的唯一标识或命名。用于批量匹配和替换所有名字符合某种规则的节点。
    """
    def __init__(self, op=None, name=None, to=None):
        # op: 正则表达式，匹配操作类型（如op_name）
        # name: 正则表达式，匹配节点名
        # to: 替换后的字符串
        # 下面三行是参数有效性检查
        assert op or name, "Either op or name must be provided"                         # 必须提供op或name之一
        assert not(op and name), "Either op or name should be provided, but not both"   # 不能同时提供op和name
        assert bool(to), "The to parameter is required"                                 # 必须提供to参数
        self.to = to
        self.op = re.compile(op) if op else None                                        # 如果有op，编译为正则表达式对象
        self.name = re.compile(name) if name else None                                  # 如果有name，编译为正则表达式对象
    
    def apply(self, graph):
        """
        遍历所有节点，根据正则表达式批量替换节点的op_name。
        """
        for i, node in enumerate(graph.nodes.values()):
            if self.op:
                # 如果设置了op正则表达式，则对op_name进行替换
                node.op_name = self.op.sub(self.to, node.op_name)
            if self.name is None:
                # 如果没有设置name正则表达式，确保op_name为字符串
                node.op_name = str(node.op_name)
            else:
                # 如果设置了name正则表达式，则对op_name进行替换
                node.op_name = self.name.sub(self.to, node.op_name)

class Fold():
    """
    节点折叠（融合）变换类。
    用于将一组满足特定模式的节点合并为一个新节点，实现结构简化或算子融合。
    """
    def __init__(self, pattern, to, name=None):
        # pattern: 匹配的节点操作序列模式（如"conv > relu"）
        # to: 新节点的操作名或特殊标记（如"__first__"表示用第一个节点替换）
        # name: 新节点的名字（可选）
        self.pattern = ge.GEParser(pattern).parse()
        self.to = to
        self.name = name

    def apply(self, graph):     
        """
        在计算图中查找所有匹配pattern的节点序列，并将其合并为一个节点。
        """
        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break

            # 根据to参数决定新节点的来源
            if self.to == "__first__":
                combo = matches[0]
            elif self.to == "__last__":
                combo = matches[-1]
            else:
                # 找到所有输出节点（即不在匹配序列中的后继节点）
                outputs = set()
                match_ids = [node.id for node in matches]
                for match_node in matches:
                    for outgoing_node in graph.outgoing(match_node):
                        if outgoing_node.id not in match_ids:
                            outputs.add(outgoing_node)
                # 合并所有操作
                combo_op = matches[0].op
                for i in range(1, len(matches)):
                    combo_op += matches[i].op
                combo_op.name = self.to or self.pattern
                # 创建新节点，输出shape取最后一个节点
                combo = Node(id=graph.sequence_id(),
                             op=combo_op,
                             output_shape=matches[-1].output_shape,
                             outputs = list(outputs)) # TODO, check bugs
                # 合并caption信息
                combo._caption = "/".join(filter(None, [l.caption for l in matches]))
            # 用新节点替换原有节点序列
            graph.replace(matches, combo)


class ConvBNFuse():
    """
    卷积+BN融合变换类。
    用于查找conv > batchnorm结构，并做标记（实际融合逻辑需在其它地方实现）。
    """
    def __init__(self, pattern, to, name=None):
        # pattern: 匹配模式（如"conv > batchnorm"）
        # to: 新操作名（如"convbn"）
        # name: 新节点名（可选）
        self.pattern = ge.GEParser(pattern).parse()
        self.to = to
        self.name = name

    def apply(self, graph):     
        """
        查找所有conv > batchnorm结构，并将其加入fused_conv_bns列表，做融合标记。
        """
        graph.fused_conv_bns = list()
        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            for match_node in matches:
                match_node._skip_pattern_search = True
            graph.fused_conv_bns.append(matches)

# PyTorch Graph Transforms
FRAMEWORK_TRANSFORMS = [
    # 将ONNX导出的操作名如"onnx::Conv"重命名为"Conv"
    Rename(op=r"onnx::(.*)", to=r"\1"),
    # 将"gemm"重命名为"linear"
    Rename(op=r"gemm", to=r"linear"),
    # 将"batchnormalization"重命名为"batchnorm"
    Rename(op=r"batchnormalization", to="batchnorm"),
]

# 卷积+BN融合变换对象（可用于进一步优化）
CONV_BN_FUSE = ConvBNFuse("conv > batchnorm", "convbn")
