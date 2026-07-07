# 引入IntEnum用于创建整数枚举类型
from enum import IntEnum

# 定义张量变换类型的枚举类，用于标识不同的变换操作
class TensorTransform(IntEnum):
    NO_UPDATE = 0                               # 不进行任何更新操作
    NO_PRUNE = 1                                # 不进行剪枝操作
    BASIC = 2                                   # 基本变换操作
    ACCESSORY = 3                               # 附属变换操作
    MULTIHEAD_HEADDIM = 4                       # 多头注意力中对头维度的变换，仅影响张量本身
    MULTIHEAD_NUMHEAD = 5                       # 多头注意力中对头数量的变换，仅影响张量本身
    REVERSE_MULTIHEAD_HEADDIM = 6               # 多头注意力中头维度的反向变换，仅影响张量本身
    REVERSE_MULTIHEAD_NUMHEAD = 7               # 多头注意力中头数量的反向变换，仅影响张量本身
    AUXILIARY = 8                               # 辅助变换操作
    TRANSPOSE = 9                               # 转置变换操作
    MULTIHEAD_NUMHEAD_SPREAD = 10               # 多头注意力中头数量的扩散变换，会影响同一节点组中的其他节点
    REVERSE_MULTIHEAD_NUMHEAD_SPREAD = 11       # 多头注意力中头数量的反向扩散变换，会影响同一节点组中的其他节点

    TOTAL = 12                                  # 变换类型的总数

# 判断给定的变换类型是否为扩散型变换（影响同组其他节点的变换）
def is_spread_transformation(transformation_type):
    if transformation_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
        return True
    elif transformation_type == TensorTransform.REVERSE_MULTIHEAD_NUMHEAD_SPREAD:
        return True
    else:
        return False

# 扩散型变换到标准变换的映射字典，用于将扩散型变换关联到对应的基本变换类型
SPREAD_TRANSFORM_MAP = {
    TensorTransform.MULTIHEAD_NUMHEAD_SPREAD: TensorTransform.MULTIHEAD_NUMHEAD
}

# 主要的张量变换函数，根据变换类型对张量进行相应的变换
# 参数:
#   tensor: 要变换的张量
#   transformation_type: 变换类型
#   num_groups: 组数量
#   num_heads: 头数量
#   head_dim: 头维度大小
def tensor_transformation(tensor, transformation_type, num_groups=1, num_heads=1, head_dim=1):
    # 对于无更新或无剪枝操作，直接返回原始张量
    if transformation_type == TensorTransform.NO_UPDATE or \
       transformation_type == TensorTransform.NO_PRUNE:
        return tensor 
    # 基本变换
    elif transformation_type == TensorTransform.BASIC:
        return basic_transformation(tensor, num_groups)
    # 附属变换，使用与基本变换相同的函数
    elif transformation_type == TensorTransform.ACCESSORY:
        return basic_transformation(tensor, num_groups)
    # 多头注意力头维度变换
    elif transformation_type == TensorTransform.MULTIHEAD_HEADDIM:
        return multihead_headdim_transformation(tensor, num_groups, num_heads)
    # 多头注意力头数量变换
    elif transformation_type == TensorTransform.MULTIHEAD_NUMHEAD:
        return multihead_numhead_transformation(tensor, num_groups)
    # 多头注意力头数量扩散变换，使用与普通头数量变换相同的函数
    elif transformation_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
        return multihead_numhead_transformation(tensor, num_groups)
    # 多头注意力头维度的反向变换
    elif transformation_type == TensorTransform.REVERSE_MULTIHEAD_HEADDIM:
        return reverse_multihead_headdim_transformation(tensor, num_groups, num_heads)
    # 多头注意力头数量的反向变换
    elif transformation_type == TensorTransform.REVERSE_MULTIHEAD_NUMHEAD:
        return reverse_multihead_numhead_transformation(tensor, num_groups, head_dim)
    # 转置变换
    elif transformation_type == TensorTransform.TRANSPOSE:
        return transpose_transformation(tensor, num_groups)
    
# 
def basic_transformation(tensor, num_groups=1):
    '''
    # 定义一个名为 basic_transformation 的函数，它接受两个参数：
    #   - tensor: 必需参数，代表输入的 PyTorch 张量，将对其进行变换。
    #   - num_groups: 可选参数，代表变换后张量的第一个维度的大小（组的数量）。如果调用函数时不提供此参数，它将默认为 1。
    # 基本变换函数：将张量重塑为 [num_groups, -1] 的形状，-1 表示自动计算该维度大小以保持元素总数不变。
    # 调用输入张量 tensor 的 view() 方法，并返回结果。
    # view() 方法用于改变张量的形状（维度和大小），但要求张量在内存中是连续的，
    # 并且新旧形状的总元素数量必须相同。
    # 参数传递给 view():
    #   - num_groups: 指定变换后张量的第一个维度的大小。
    #   - -1: 这是一个特殊值，告诉 view() 方法自动推断该维度的大小，
    #         以确保变换后的张量总元素数量与原始张量相同。
    #         例如，如果原始张量有 12 个元素，num_groups 是 2，
    #         则 -1 会被推断为 6，使得新形状为 (2, 6)。
    #         如果原始张量有 12 个元素，num_groups 是 1，
    #         则 -1 会被推断为 12，使得新形状为 (1, 12)。
    # 最终效果：将输入张量重塑为一个二维张量，其中第一维的大小是 num_groups，
    # 第二维的大小是根据原始张量总元素数量和 num_groups 自动计算得到的。
    '''
    return tensor.view(num_groups, -1)

# 多头注意力头维度变换：
# 1. 将张量重塑为 [num_heads, num_groups, -1]
# 2. 置换维度顺序为 [num_groups, num_heads, -1]
# 3. 确保内存连续，再重塑为 [num_groups, -1]
def multihead_headdim_transformation(tensor, num_groups=1, num_heads=1):
    return tensor.view(num_heads, num_groups, -1).permute(1, 0, 2).contiguous().view(num_groups, -1)

# 多头注意力头数量变换：将张量重塑为 [num_groups, -1]
# 注意：这里使用reshape而不是view，可能是为了处理非连续内存的情况
def multihead_numhead_transformation(tensor, num_groups=1):
    return tensor.reshape(num_groups, -1)

# 多头注意力头维度的反向变换
# 如果张量元素数量足够多（大于等于num_groups * num_heads）:
#   1. 将张量重塑为 [num_groups, num_heads, -1]
#   2. 置换维度顺序为 [num_heads, num_groups, -1]
#   3. 确保内存连续，再重塑为 [num_heads * num_groups, -1]
# 否则，对于一维张量:
#   进行特殊处理，先扩展维度，然后重复数据，再进行维度变换
def reverse_multihead_headdim_transformation(tensor, num_groups=1, num_heads=1):
    if tensor.numel() >= num_groups * num_heads:
        return tensor.view(num_groups, num_heads, -1).permute(1, 0, 2).contiguous().view(num_heads * num_groups, -1)
    else:
        if len(tensor.shape) == 1:
            return tensor.unsqueeze(1).repeat(1, num_heads).view(num_groups, num_heads, -1).permute(1, 0, 2).contiguous()\
                    .view(num_heads * num_groups, -1).squeeze()
        else:
            return tensor

# 多头注意力头数量的反向变换
# 如果张量元素数量足够多（大于等于num_groups * head_dim）:
#   目前尚未实现，会抛出NotImplementedError异常
# 否则，对于一维张量:
#   进行特殊处理，先扩展维度，然后重复数据，再重塑为所需形状
def reverse_multihead_numhead_transformation(tensor, num_groups=1, head_dim=1):
    if tensor.numel() >= num_groups * head_dim:
        raise NotImplementedError
    else:
        if len(tensor.shape) == 1:
            return tensor.unsqueeze(1).repeat(1, head_dim).view(num_groups * head_dim, -1).squeeze()
        else:
            return tensor
        
# 转置变换，根据张量的维度数量采取不同的处理:
# - 一维张量: 重塑为 [num_groups, -1]
# - 二维张量: 先转置维度，再确保内存连续，最后重塑为 [num_groups, -1]
# - 四维张量: 先对前两个维度进行转置，再确保内存连续，最后重塑为 [num_groups, -1]
def transpose_transformation(tensor, num_groups=1):
    if len(tensor.shape) == 1:
        return tensor.view(num_groups, -1)
    elif len(tensor.shape) == 2:
        return tensor.permute(1, 0).contiguous().view(num_groups, -1)
    elif len(tensor.shape) == 4:
        return tensor.permute(1, 0, 2, 3).contiguous().view(num_groups, -1)
    