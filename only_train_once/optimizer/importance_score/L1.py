import torch
from only_train_once.transform import tensor_transformation, TensorTransform


def importance_score_by_l1_magnitude(param_group):
    """
    计算参数组内所有参数（经过变换后）的 L1 范数作为重要性分数 (DHSPG 策略)。
    该函数处理参数组内的所有参数，对它们进行可能的变换（如分组），然后计算变换后的参数在结构单元维度上的 L1 范数。
    分数最终存储在 param_group['importance_scores']['l1_magnitude'] 中。
    Args:
        param_group (dict): 优化器中的参数组字典，包含基本配置信息。
    """
    norm_group = None
    for param, p_transform in zip(param_group['params'], param_group['p_transform']):
        param_transform = None
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])

        # 直接计算L1范数(p=1)，不需要平方和平方根操作
        if norm_group == None:
            norm_group = torch.norm(param_transform, dim=1, p=1)
        else:
            norm_group += torch.norm(param_transform, dim=1, p=1)
    
    param_group['importance_scores']['l1_magnitude'] = norm_group

def importance_score_by_l1_avg_magnitude(param_group):
    """
    计算参数组内所有参数（经过变换后）的平均 L1 范数作为重要性分数 (DHSPG 策略)。
    与 l1_magnitude 类似，但最后会除以变换后结构单元的总大小。
    分数最终存储在 param_group['importance_scores']['l1_avg_magnitude'] 中。
    """
    norm_group = None
    group_sizes = 0
    for param, p_transform in zip(param_group['params'], param_group['p_transform']):
        param_transform = None
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
            
        if norm_group == None:
            norm_group = torch.norm(param_transform, dim=1, p=1)
        else:
            norm_group += torch.norm(param_transform, dim=1, p=1)
        
        group_sizes += param_transform.shape[1]
    
    param_group['importance_scores']['l1_avg_magnitude'] = norm_group / float(group_sizes + 1e-6)

def importance_score_by_l1_magnitude_lora(param_group):
    """
    计算参数组内所有参数（经过变换后，排除 LoRA 参数）的 L1 范数作为重要性分数 (LHSPG 策略)。
    与 DHSPG 版本类似，但会跳过名称中包含 'lora_A' 或 'lora_B' 的参数。
    分数最终存储在 param_group['importance_scores']['l1_magnitude'] 中。
    """
    norm_group = None
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        if 'lora_A' in p_name or 'lora_B' in p_name:
            continue
            
        param_transform = None
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
            
        if norm_group == None:
            norm_group = torch.norm(param_transform, dim=1, p=1)
        else:
            norm_group += torch.norm(param_transform, dim=1, p=1)
    
    param_group['importance_scores']['l1_magnitude'] = norm_group

def importance_score_by_l1_avg_magnitude_lora(param_group):
    """
    计算参数组内所有参数（经过变换后，排除 LoRA 参数）的平均 L1 范数作为重要性分数 (LHSPG 策略)。
    与 LHSPG l1_magnitude 类似，但最后会除以变换后结构单元的总数量。
    分数最终存储在 param_group['importance_scores']['l1_avg_magnitude'] 中。
    """
    norm_group = None
    group_sizes = 0
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        if 'lora_A' in p_name or 'lora_B' in p_name:
            continue
            
        param_transform = None
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])
            
        if norm_group == None:
            norm_group = torch.norm(param_transform, dim=1, p=1)
        else:
            norm_group += torch.norm(param_transform, dim=1, p=1)
        
        group_sizes += param_group['num_groups']
    
    param_group['importance_scores']['l1_avg_magnitude'] = norm_group / float(group_sizes + 1e-6)
