import torch
from only_train_once.transform import tensor_transformation, TensorTransform

LORA_NAMES = ['lora_B', 'lora_A', 'lora_embedding_B', 'lora_embedding_A']


def importance_score_by_l2_magnitude(param_group):
    """
    计算参数组内所有参数（经过变换后）的 L2 范数作为重要性分数 (DHSPG 策略)。
    该函数处理参数组内的所有参数，对它们进行可能的变换（如分组），然后计算变换后的参数在结构单元维度上的 L2 范数。
    分数最终存储在 param_group['importance_scores']['magnitude'] 中。
    Args:
        param_group (dict): 优化器中的参数组字典，包含:
            - 'params': 参数张量列表。
            - 'p_transform': 每个参数对应的变换类型列表。
            - 'num_groups': 分组数量（如果使用分组变换）。
            - 'num_heads': 注意力头数（如果使用多头注意力变换）。
            - 'importance_scores': 将存储计算出的重要性分数的字典。
    """
    norm_group = None       # 初始化累积参数范数的平方和为 None
    # 迭代参数名称、参数张量和对应的变换类型
    for param, p_transform in zip(param_group['params'], param_group['p_transform']):
        param_transform = None      # 初始化变换后的参数张量
        # 检查参数的变换类型是否为多头注意力头维度
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 如果是，使用指定的分组数和头数对参数张量进行变换
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 如果不是，使用指定的分组数进行变换 (适用于其他类型的变换)
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])

        # 累积计算变换后参数的 L2 范数的平方和，L2 范数在 dim=1 (结构单元维度) 上计算
        if norm_group == None: # 检查是否是处理的第一个非 LoRA 参数
            # 如果是第一个，直接将当前参数变换后的范数平方赋值给 norm_group
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            # 如果不是第一个，将当前参数变换后的范数平方累加到 norm_group
            norm_group += torch.norm(param_transform, dim=1) ** 2
    # 计算所有非 LoRA 参数累积范数平方和的平方根，得到最终的 L2 范数
    # 并将结果存储在参数组字典的 'importance_scores' 下的 'magnitude' 键中
    param_group['importance_scores']['l2_magnitude'] = torch.sqrt(norm_group)

def importance_score_by_l2_avg_magnitude(param_group):
    """
    计算参数组内所有参数（经过变换后）的 平均 L2 范数 作为重要性分数 (DHSPG 策略)。
    与 magnitude 类似，但最后会除以变换后结构单元的总大小。
    分数最终存储在 param_group['importance_scores']['avg_magnitude'] 中。
    Args:
        param_group (dict): 优化器中的参数组字典 (同上)。
    """
    # 初始化累积参数范数的平方和为 None
    norm_group = None
    # 初始化变换后结构单元的总大小计数器
    group_sizes = 0
    # 迭代参数张量和对应的变换类型
    for param, p_transform in zip(param_group['params'], param_group['p_transform']):
        # 初始化变换后的参数张量
        param_transform = None
        # 如果变换类型是多头注意力头维度
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 使用指定的分组数和头数进行变换
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 使用指定的分组数进行变换 (适用于其他类型的变换)
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
        # 累积计算变换后参数的 L2 范数的平方和
        if norm_group == None: # 检查是否是处理的第一个参数
            # 如果是第一个，直接将当前参数变换后的范数平方赋值给 norm_group
            # torch.norm(param_transform, dim=1) 计算每个结构单元的 L2 范数
            # ** 2 计算范数的平方
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            # 如果不是第一个，将当前参数变换后的范数平方累加到 norm_group
            norm_group += torch.norm(param_transform, dim=1) ** 2
        # 累加当前变换后参数的结构单元大小
        # param_transform.shape[1] 是变换后每个结构单元包含的元素数量 (即每个组的大小)
        group_sizes += param_transform.shape[1]
    # 计算最终的 L2 范数 (累积平方和的平方根)，然后除以总大小 (group_sizes) 得到平均幅值
    # 使用 float() 确保是浮点数除法，添加 1e-6 是为了防止 group_sizes 为零导致除以零错误，提高数值稳定性
    # 将结果存储在参数组字典的 'importance_scores' 下的 'avg_magnitude' 键中
    param_group['importance_scores']['l2_avg_magnitude'] = torch.sqrt(norm_group) / float(group_sizes + 1e-6)

def importance_score_by_l2_magnitude_lora(param_group):
    """
    计算参数组内所有参数（经过变换后，排除 LoRA 参数）的 L2 范数作为重要性分数 (LHSPG 策略)。
    与 DHSPG 版本类似，但会跳过名称中包含 'lora_A' 或 'lora_B' 的参数。
    分数最终存储在 param_group['importance_scores']['magnitude'] 中。
    Args:
        param_group (dict): 优化器中的参数组字典，包含:
            - 'p_names': 参数名称列表。
            - 'params': 参数张量列表。
            - 'p_transform': 每个参数对应的变换类型列表。
            - 'num_groups': 分组数量。
            - 'num_heads': 注意力头数。
            - 'importance_scores': 存储分数的字典。
    """
    norm_group = None       # 初始化累积参数范数的平方和为 None，用于存储所有非 LoRA 参数的范数平方和
    # 迭代参数名称、参数张量和对应的变换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        # 检查当前参数名称是否包含 'lora_A' 或 'lora_B'
        if any(lora_name in p_name for lora_name in LORA_NAMES):
            # 如果是 LoRA 相关参数，则跳过，不参与幅值重要性计算
            continue # 直接进入下一次循环
        param_transform = None      # 初始化用于存储变换后参数的变量
        # 检查当前参数的变换类型是否为多头注意力头维度
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 如果是，调用 tensor_transformation 函数，使用指定的分组数和头数对参数进行变换
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 如果不是多头注意力类型，调用 tensor_transformation 函数，仅使用指定的分组数进行变换
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
        # 累积计算变换后参数的 L2 范数的平方和，检查 norm_group 是否仍为 None，即是否是第一个被处理的非 LoRA 参数
        if norm_group == None:
            # 如果是第一个非 LoRA 参数，直接计算其变换后张量在维度 1 上的 L2 范数的平方，并赋值给 norm_group
            # torch.norm(param_transform, dim=1) 计算沿维度 1 (结构单元维度) 的 L2 范数，** 2 计算范数的平方
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            # 如果不是第一个非 LoRA 参数，将当前参数变换后的范数平方累加到已有的 norm_group 上
            norm_group += torch.norm(param_transform, dim=1) ** 2
    # 循环结束后，norm_group 包含了所有非 LoRA 参数变换后的范数平方之和，计算这个累积平方和的平方根，得到最终的重要性分数 (整体 L2 范数)
    # 将计算出的幅值重要性分数存储到参数组字典的 'importance_scores' 条目下的 'magnitude' 键中
    param_group['importance_scores']['l2_magnitude'] = torch.sqrt(norm_group)

def importance_score_by_l2_avg_magnitude_lora(param_group):
    """
    计算参数组内所有参数（经过变换后，排除 LoRA 参数）的平均 L2 范数作为重要性分数 (LHSPG 策略)。
    与 LHSPG magnitude 类似，但最后会除以变换后结构单元的总数量 (这里用 num_groups 近似)。
    分数最终存储在 param_group['importance_scores']['avg_magnitude'] 中。
    Args:
        param_group (dict): 优化器中的参数组字典 (同上)。
    """
    norm_group = None   # 初始化累积参数范数的平方和为 None
    group_sizes = 0     # 初始化变换后结构单元的总数量计数器
    # 迭代参数名称、参数张量和对应的变换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        # 如果参数名称包含 'lora_A' 或 'lora_B'，则跳过该参数
        if any(lora_name in p_name for lora_name in LORA_NAMES):
            continue # 继续下一次循环迭代
        param_transform = None      # 初始化变换后的参数张量
        # 检查参数的变换类型是否为多头注意力头维度
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 如果是，使用指定的分组数和头数进行变换
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 如果不是，使用指定的分组数进行变换
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
        # 累积计算变换后参数的 L2 范数的平方和
        if norm_group == None: # 检查是否是处理的第一个非 LoRA 参数
            # 如果是第一个非 LoRA 参数，直接计算其变换后张量在维度 1 上的 L2 范数的平方，并赋值给 norm_group
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            # 如果不是第一个非 LoRA 参数，将当前参数变换后的范数平方累加到已有的 norm_group 上
            norm_group += torch.norm(param_transform, dim=1) ** 2
        # 累加当前参数对应的分组数量 (注意：这里直接累加了 param_group['num_groups']，
        # 这假设每个参数变换后都产生 num_groups 个单元，可能与 DHSPG 版本计算 group_sizes 的方式不同)
        group_sizes += param_group['num_groups']
    # 计算最终的 L2 范数 (平方根)，然后除以累积的组大小得到平均幅值，使用 float() 确保除法是浮点数除法
    # 添加 1e-6 是为了防止 group_sizes 为零导致除以零错误，提高数值稳定性
    param_group['importance_scores']['l2_avg_magnitude'] = torch.sqrt(norm_group) / float(group_sizes + 1e-6)
