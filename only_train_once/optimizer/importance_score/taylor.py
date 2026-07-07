import torch
import torch.nn.functional as F
from only_train_once.transform import tensor_transformation, TensorTransform

LORA_NAMES = [('lora_B', 'lora_A'), ('lora_embedding_B', 'lora_embedding_A')]

def importance_score_by_first_order_taylor(param_group):
    """
    计算基于 一阶泰勒展开的重要性分数 (DHSPG - Direct Hessian-vector Product approximation using Stochastic Polyak Gradients)。
    该方法使用参数与其梯度变体（例如，Polyak平均梯度）的内积来近似重要性。
    Args:
        param_group (dict): 包含参数、梯度变体、转换信息等的参数组字典。
    """
    params_grads_inner_prod = None      # 初始化参数和梯度变体的内积总和
    # 遍历参数组中的每个参数名称、参数张量和参数转换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        # 如果当前参数没有对应的梯度变体，则跳过
        if p_name not in param_group['grad_variant']:
            continue
        # 获取当前参数对应的梯度变体
        grad = param_group['grad_variant'][p_name]
        param_transform = None      # 初始化参数转换后的张量
        # 根据指定的转换类型对参数张量进行转换
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 如果是多头注意力且按头维度转换
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 其他转换类型
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])

        grad_transform = None       # 初始化梯度转换后的张量
        # 根据指定的转换类型对梯度张量进行转换
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 如果是多头注意力且按头维度转换
            grad_transform = tensor_transformation(grad, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 其他转换类型
            grad_transform = tensor_transformation(grad, p_transform, param_group['num_groups'])

        # 累加内积
        if params_grads_inner_prod == None:
            # 如果是第一个参数，直接赋值
            params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
        else:
            # 否则累加到总和中
            params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)
    # 将计算得到的内积的绝对值作为一阶泰勒重要性分数存储到参数组中
    param_group['importance_scores']['taylor_first_order'] = torch.abs(params_grads_inner_prod)

def importance_score_by_second_order_taylor(param_group):
    """
    计算基于二阶泰勒展开的重要性分数 (DHSPG)。
    该方法利用一阶泰勒分数（如果已计算）或重新计算参数与梯度变体的内积来近似二阶重要性。
    Args:
        param_group (dict): 包含参数、梯度变体、转换信息等的参数组字典。
    """
    # 检查是否已经计算过一阶泰勒重要性分数
    if 'taylor_first_order' in param_group['importance_scores']:
        # 如果已存在，直接使用一阶分数的平方乘以0.5作为二阶分数（基于泰勒展开的近似）
        param_group['importance_scores']['taylor_second_order'] = 0.5 * param_group['importance_scores']['taylor_first_order'] ** 2
        # 计算完成，直接返回
        return

    # 如果一阶分数不存在，则重新计算参数和梯度变体的内积
    params_grads_inner_prod = None
    # 遍历参数组中的每个参数名称、参数张量和参数转换类型
    # 注意：原始代码中注释掉了基于 zip(param_group['params'], param_group['grad_variant'], param_group['p_transform']) 的迭代方式
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        # 如果当前参数没有对应的梯度变体，则跳过
        if p_name not in param_group['grad_variant']:
            continue
        grad = param_group['grad_variant'][p_name]      # 获取当前参数对应的梯度变体
        param_transform = None                          # 初始化参数转换后的张量
        # 根据指定的转换类型对参数张量进行转换
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])

        
        grad_transform = None       # 初始化梯度转换后的张量
        # 根据指定的转换类型对梯度张量进行转换
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            grad_transform = tensor_transformation(grad, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            grad_transform = tensor_transformation(grad, p_transform, param_group['num_groups'])

        # 累加内积
        if params_grads_inner_prod == None:
            params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
        else:
            params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)
    # 将计算得到的内积的平方乘以0.5作为二阶泰勒重要性分数存储到参数组中
    param_group['importance_scores']['taylor_second_order'] = 0.5 * params_grads_inner_prod ** 2

def importance_score_by_first_order_taylor_lora(param_group, global_params):
    """
    计算基于一阶泰勒展开的重要性分数 (LHSPG - LoRA-based Hessian-vector Product approximation using Stochastic Polyak Gradients)。
    该方法使用原始参数与LoRA参数（B和A的乘积）计算得到的“梯度近似”的内积来近似重要性。
    Args:
        param_group (dict): 包含LoRA参数、转换信息等的参数组字典。
        global_params (dict): 包含模型所有参数（包括原始参数和LoRA参数）的字典。
    """
    # 初始化原始参数和LoRA梯度近似的内积总和
    params_grads_inner_prod = None
    # 遍历参数组中的每个参数名称、参数张量（这里主要是LoRA参数）和参数转换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        for lora_strs in LORA_NAMES:
            if lora_strs[0] in p_name:
                lora_A_name = p_name.replace(lora_strs[0], lora_strs[1])
                lora_A = global_params[lora_A_name]
                lora_BA = torch.matmul(param, lora_A)
                original_param_name = p_name.split(lora_strs[0])[0] + 'weight'
                original_param = global_params[original_param_name]

                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'], param_group['num_heads'])
                elif lora_strs[0] == 'lora_embedding_B':
                    param_transform = tensor_transformation(original_param, TensorTransform.TRANSPOSE, param_group['num_groups'])
                else:
                    param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'])
                grad_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'], param_group['num_heads'])
                else:
                    grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'])

            # 累加内积
            if params_grads_inner_prod == None:
                params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
            else:
                params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)

    # 将计算得到的内积的绝对值作为一阶泰勒重要性分数存储到参数组中
    param_group['importance_scores']['taylor_first_order'] = torch.abs(params_grads_inner_prod)

def importance_score_by_second_order_taylor_lora(param_group, global_params):
    """
    计算基于二阶泰勒展开的重要性分数 (LHSPG)。
    该方法利用一阶泰勒分数（如果已计算）或重新计算原始参数与LoRA梯度近似的内积来近似二阶重要性。
    Args:
        param_group (dict): 包含LoRA参数、转换信息等的参数组字典。
        global_params (dict): 包含模型所有参数（包括原始参数和LoRA参数）的字典。
    """
    # 检查是否已经计算过一阶泰勒重要性分数
    if 'taylor_first_order' in param_group['importance_scores']:
        # 如果已存在，直接使用一阶分数的平方乘以0.5作为二阶分数
        param_group['importance_scores']['taylor_second_order'] = 0.5 * param_group['importance_scores']['taylor_first_order'] ** 2
        # 计算完成，直接返回
        return

    # 如果一阶分数不存在，则重新计算原始参数和LoRA梯度近似的内积
    params_grads_inner_prod = None
    # 遍历参数组中的每个参数名称、参数张量（这里主要是LoRA参数）和参数转换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        for lora_strs in LORA_NAMES:
            if lora_strs[0] in p_name:
                lora_A_name = p_name.replace(lora_strs[0], lora_strs[1])
                lora_A = global_params[lora_A_name]
                lora_BA = torch.matmul(param, lora_A)
                original_param_name = p_name.split(lora_strs[0])[0] + 'weight'
                original_param = global_params[original_param_name]

                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'], param_group['num_heads'])
                elif lora_strs[0] == 'lora_embedding_B':
                    param_transform = tensor_transformation(original_param, TensorTransform.TRANSPOSE, param_group['num_groups'])
                else:
                    param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'])
                grad_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'], param_group['num_heads'])
                else:
                    grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'])


            # 累加内积
            if params_grads_inner_prod == None:
                params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
            else:
                params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)

    # 将计算得到的内积的平方乘以0.5作为二阶泰勒重要性分数存储到参数组中
    param_group['importance_scores']['taylor_second_order'] = 0.5 * params_grads_inner_prod ** 2
