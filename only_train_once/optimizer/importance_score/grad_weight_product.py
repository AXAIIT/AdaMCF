import torch
from only_train_once.transform import tensor_transformation, TensorTransform

def importance_score_by_grad_weight_product(param_group):
    """
    计算参数的梯度与权重乘积的L1范数，作为重要性分数 (DHSPG策略)。
    这个标准结合了参数的大小和梯度的大小，认为绝对值大且梯度也大的参数更重要。
    """
    # 初始化用于累加整个参数组重要性分数的变量
    score_group = None
    # 遍历参数组中的每一个参数及其名称和变换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        # 检查是否存在对应的平滑梯度 (grad_variant)。如果没有，则跳过该参数。
        # 'grad_variant' 是预先计算并存储在 param_group 中的平滑梯度。
        if p_name not in param_group['grad_variant']:
            continue
        
        # 从 'grad_variant' 字典中获取平滑梯度，而不是使用原始的 param.grad
        grad = param_group['grad_variant'][p_name]
        # 计算重要性：参数值和其梯度的逐元素乘积的绝对值。
        # 这反映了参数值和梯度大小的综合影响。
        importance = torch.abs(param.data * grad.data)
        
        # 根据参数的变换类型（p_transform）对重要性张量进行重塑，以便按结构（如神经元、通道）进行求和。
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 针对多头注意力机制的特殊处理，需要额外的头数量(num_heads)信息。
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
            
    # 将最终计算出的整个参数组的重要性分数存储回 param_group 字典中。
    param_group['importance_scores']['grad_weight_product'] = score_group

def importance_score_by_grad_weight_product_lora(param_group):
    """
    计算参数的梯度与权重乘积的L1范数，作为重要性分数 (LHSPG策略, 跳过LoRA)。
    注意：LHSPG 策略通常不直接使用 grad_variant，此函数为保持API一致性。
    在LHSPG中，梯度信息由LoRA矩阵的乘积隐式提供，因此这个标准可能不适用于LHSPG。
    这里我们保留实现，但假设 grad_variant 存在。
    """
    # 初始化用于累加整个参数组重要性分数的变量
    score_group = None
    # 遍历参数组中的每一个参数及其名称和变换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        # LHSPG策略中，需要跳过LoRA相关的参数('lora_A', 'lora_B')。
        # 同时，也检查平滑梯度是否存在。
        if 'lora_A' in p_name or 'lora_B' in p_name or p_name not in param_group.get('grad_variant', {}):
            continue
            
        # 从 'grad_variant' 字典中获取平滑梯度
        grad = param_group['grad_variant'][p_name]
        # 计算重要性：参数值和其梯度的逐元素乘积的绝对值
        importance = torch.abs(param.data * grad.data)
        
        # 根据参数的变换类型对重要性张量进行重塑
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 针对多头注意力机制的特殊处理
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
            
    # 将最终计算出的整个参数组的重要性分数存储回 param_group 字典中
    param_group['importance_scores']['grad_weight_product'] = score_group

