import torch
from only_train_once.transform import tensor_transformation, TensorTransform

def importance_score_by_fisher(param_group):
    """
    计算Fisher信息对角线的L1范数 (梯度的平方)，作为重要性分数 (DHSPG策略)。
    Fisher信息矩阵的对角线元素可以近似为梯度的平方，它衡量了参数对模型输出（或损失）的敏感度。
    分数越高，表示参数越重要。
    """
    # 初始化用于累加整个参数组重要性分数的变量
    score_group = None
    # 遍历参数组中的每一个参数、其名称以及其变换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        # 检查是否存在对应的平滑梯度 (grad_variant)。如果不存在，则跳过该参数。
        # 'grad_variant' 是预先计算并存储的平滑梯度。
        if p_name not in param_group['grad_variant']:
            continue
        
        # 从 'grad_variant' 字典中获取平滑梯度
        grad = param_group['grad_variant'][p_name]
        # 计算重要性分数：梯度的平方。这是Fisher信息对角线的近似。
        importance = grad.data ** 2
        
        # 根据参数的变换类型（p_transform）对重要性张量进行重塑，以便按结构（如神经元、通道）进行求和。
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 对多头注意力机制进行特殊处理，需要额外的头数量(num_heads)信息。
            param_transform = tensor_transformation(importance, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 应用通用的张量变换。
            param_transform = tensor_transformation(importance, p_transform, param_group['num_groups'])
            
        # 沿维度1对变换后的重要性张量求和，得到每个可剪枝单元（如一个神经元或一个卷积核）的总重要性分数。
        current_score = torch.sum(param_transform, dim=1)
        # 将当前参数的重要性分数累加到整个组的分数上。
        if score_group is None:
            # 如果是组里的第一个参数，直接赋值。
            score_group = current_score
        else:
            # 否则，与之前的分数相加。
            score_group += current_score
            
    # 将最终计算出的整个参数组的重要性分数存储回 param_group 字典中，键为 'fisher'。
    param_group['importance_scores']['fisher'] = score_group

def importance_score_by_fisher_lora(param_group):
    """
    计算Fisher信息对角线的L1范数 (梯度的平方)，作为重要性分数 (LHSPG策略, 跳过LoRA)。
    同样，此标准可能不直接适用于LHSPG，但为保持API一致性而实现。
    LHSPG策略通常关注原始权重，而Fisher信息依赖于梯度，这里的实现提供了一种可能的评估方式。
    """
    # 初始化用于累加整个参数组重要性分数的变量
    score_group = None
    # 遍历参数组中的每一个参数、其名称以及其变换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        # 对于LHSPG策略，跳过LoRA相关的参数('lora_A', 'lora_B')。
        # 同时，也检查平滑梯度是否存在。
        if 'lora_A' in p_name or 'lora_B' in p_name or p_name not in param_group.get('grad_variant', {}):
            continue
            
        # 从 'grad_variant' 字典中获取平滑梯度
        grad = param_group['grad_variant'][p_name]
        # 计算重要性分数：梯度的平方
        importance = grad.data ** 2
        
        # 根据参数的变换类型对重要性张量进行重塑
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 对多头注意力机制进行特殊处理
            param_transform = tensor_transformation(importance, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 应用通用的张量变换
            param_transform = tensor_transformation(importance, p_transform, param_group['num_groups'])
            
        # 沿维度1对变换后的重要性张量求和，得到每个可剪枝单元的重要性分数
        current_score = torch.sum(param_transform, dim=1)
        # 将当前参数的重要性分数累加到整个组的分数上
        if score_group is None:
            score_group = current_score
        else:
            score_group += current_score
            
    # 将最终计算出的整个参数组的重要性分数存储回 param_group 字典中，键为 'fisher'。
    param_group['importance_scores']['fisher'] = score_group
