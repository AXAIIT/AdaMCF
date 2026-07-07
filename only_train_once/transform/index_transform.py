from .tensor_transform import TensorTransform


def index_transformation_param_group(indexes_by_groups, transformation_type, param_group):
    """
    根据参数组信息对索引进行转换，并确保索引合法。
    
    参数:
        indexes_by_groups: 按组划分的索引列表
        transformation_type: 转换类型
        param_group: 包含参数组信息的字典，例如 num_heads 和 head_dim
        
    返回:
        经过转换后的索引列表
    """
    if transformation_type == TensorTransform.MULTIHEAD_HEADDIM:
        refined_indexes = index_transformation(indexes_by_groups, transformation_type, num_heads=param_group['num_heads'], head_dim=param_group['head_dim'])
    elif transformation_type == TensorTransform.MULTIHEAD_NUMHEAD or transformation_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
        refined_indexes = index_transformation(indexes_by_groups, transformation_type, head_dim=param_group['head_dim'])
    else:
        refined_indexes = index_transformation(indexes_by_groups, transformation_type)

    # 确保索引合法
    max_index = param_group.get('num_groups', float('inf'))  # 获取参数组的最大索引范围
    refined_indexes = [i for i in refined_indexes if 0 <= i < max_index]

    return refined_indexes

def index_transformation(indexes_by_groups, transformation_type, num_heads=1, head_dim=1):
    """
    根据转换类型对索引进行具体处理，并确保索引合法。
    
    参数:
        indexes_by_groups: 按组划分的索引列表
        transformation_type: 转换类型
        num_heads: 多头注意力机制中的头数，默认为1
        head_dim: 每个头的维度，默认为1
        
    返回:
        经过转换后的索引列表
    """
    if transformation_type == TensorTransform.NO_UPDATE or \
       transformation_type == TensorTransform.NO_PRUNE or \
       transformation_type == TensorTransform.BASIC or \
       transformation_type == TensorTransform.ACCESSORY or \
       transformation_type == TensorTransform.TRANSPOSE:
        # 直接返回原始索引，不进行任何操作
        return indexes_by_groups 
    elif transformation_type == TensorTransform.MULTIHEAD_HEADDIM:
        # 对每个头的维度进行扩展
        refined_indexes = [i for i in indexes_by_groups]
        for h in range(1, num_heads):
            refined_indexes.extend([i + head_dim * h for i in indexes_by_groups])
        return refined_indexes
    elif transformation_type == TensorTransform.MULTIHEAD_NUMHEAD or \
         transformation_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
        # 对每个头进行索引扩展
        refined_indexes = list()
        for i in indexes_by_groups:
            refined_indexes.extend([h + i * head_dim for h in range(head_dim)])
        return refined_indexes
    else:
        # 默认情况下返回原始索引
        return indexes_by_groups
