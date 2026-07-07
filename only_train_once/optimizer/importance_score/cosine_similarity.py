'''# 导入 PyTorch 库
import torch
import torch.nn.functional as F     # 导入 PyTorch 的神经网络函数库，通常用于激活函数、损失函数等
# 从 only_train_once.transform 模块导入 tensor_transformation 函数和 TensorTransform 枚举
# tensor_transformation: 用于根据指定的转换类型重塑或变换张量
# TensorTransform: 可能是一个包含不同张量转换类型的枚举 (例如，按组、按多头注意力头等)
from only_train_once.transform import tensor_transformation, TensorTransform

LORA_NAMES = [('lora_B', 'lora_A'), ('lora_embedding_B', 'lora_embedding_A')]

# 定义一个函数，用于计算 DHSPG 策略下的基于 余弦相似度 的重要性分数
def importance_score_by_cosine_similarity(param_group):
    """
    计算参数与其对应梯度（变体）之间的余弦相似度作为重要性分数 (DHSPG 策略)。
    该函数处理参数组内的所有参数，对它们进行可能的变换（如分组），然后计算变换后的参数与其对应变换后的梯度之间的余弦相似度。
    分数最终存储在 param_group['importance_scores']['cosine_similarity'] 中。
    Args:
        param_group (dict): 优化器中的参数组字典，包含:
            - 'p_names': 参数名称列表。
            - 'params': 参数张量列表。
            - 'p_transform': 每个参数对应的变换类型列表。
            - 'grad_variant': 一个字典，存储参数名称到其梯度（或梯度变体）的映射。
            - 'num_groups': 分组数量（如果使用分组变换）。
            - 'num_heads': 注意力头数（如果使用多头注意力变换）。
            - 'importance_scores': 将存储计算出的重要性分数的字典。
    """
    norm_params = None                  # 初始化累积参数范数的平方和为 None
    norm_grads = None                   # 初始化累积梯度范数的平方和为 None
    params_grads_inner_prod = None      # 初始化累积参数和梯度内积的和为 None

    # 并行迭代参数名称、参数张量和参数变换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        # 检查当前参数名称是否存在于梯度变体字典中，如果不存在则跳过该参数
        if p_name not in param_group['grad_variant']:
            continue
        # 获取当前参数对应的梯度（或梯度变体）
        grad = param_group['grad_variant'][p_name]

        param_transform = None      # 对参数张量进行变换
        # 如果变换类型是多头注意力头维度
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 使用指定的分组数和头数进行变换
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 使用指定的分组数进行变换 (适用于其他类型的变换)
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])

        # 累积计算变换后参数的 L2 范数的平方和 (按变换后的维度 dim=1 计算范数)
        if norm_params == None:
            # 如果是第一个参数，直接赋值
            norm_params = torch.norm(param_transform, dim=1) ** 2
        else:
            # 否则，累加到之前的和上
            norm_params += torch.norm(param_transform, dim=1) ** 2

        grad_transform = None       # 对梯度张量进行与参数相同的变换
        # 如果变换类型是多头注意力头维度
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            # 使用指定的分组数和头数进行变换
            grad_transform = tensor_transformation(grad, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            # 使用指定的分组数进行变换
            grad_transform = tensor_transformation(grad, p_transform, param_group['num_groups'])

        # 累积计算变换后梯度的 L2 范数的平方和 (按变换后的维度 dim=1 计算范数)
        if norm_grads == None:
            # 如果是第一个梯度，直接赋值
            norm_grads = torch.norm(grad_transform, dim=1) ** 2
        else:
            # 否则，累加到之前的和上
            norm_grads += torch.norm(grad_transform, dim=1) ** 2

        # 累积计算变换后参数和梯度的内积 (按变换后的维度 dim=1 求和)
        if params_grads_inner_prod == None:
            # 如果是第一个，直接赋值
            params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
        else:
            # 否则，累加到之前的和上
            params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)

    # 计算累积范数平方和的平方根，得到最终的 L2 范数
    norm_params = torch.sqrt(norm_params)
    norm_grads = torch.sqrt(norm_grads)

    # 计算余弦相似度：内积 / (参数范数 * 梯度范数)
    # 添加 1e-8 是为了防止除以零，提高数值稳定性
    # 最后加 1 是将余弦相似度范围从 [-1, 1] 映射到 [0, 2]，可能便于后续处理（例如确保重要性非负）
    cosine_sim = params_grads_inner_prod / (norm_params + 1e-8) / (norm_grads + 1e-8)
    # 取绝对值，并移除 +1
    param_group['importance_scores']['cosine_similarity'] = torch.abs(cosine_sim)


# 定义一个函数，用于计算 LHSPG 策略下的基于 余弦相似度 的重要性分数
# 这个版本特别处理了 LoRA (Low-Rank Adaptation) 参数
def importance_score_by_cosine_similarity_lora(param_group, global_params):
    """
    计算 LHSPG 策略下的余弦相似度重要性分数，特别关注 LoRA 参数。
    它计算原始参数与 LoRA 更新（由 lora_B * lora_A 计算得到）之间的余弦相似度。
    分数最终存储在 param_group['importance_scores']['cosine_similarity'] 中。
    Args:
        param_group (dict): 优化器中的参数组字典，包含:
            - 'p_names': 参数名称列表。
            - 'params': 参数张量列表 (这里主要用到 lora_B)。
            - 'p_transform': 每个参数对应的变换类型列表。
            - 'num_groups': 分组数量。
            - 'num_heads': 注意力头数。
            - 'importance_scores': 将存储计算出的重要性分数的字典。
        global_params (dict): 一个包含全局参数的字典，用于查找 lora_A 和原始参数。
    """
    norm_params = None                  # 初始化累积原始参数范数的平方和为 None
    norm_grads = None                   # 初始化累积 LoRA 更新范数的平方和为 None (将 LoRA 更新视为梯度的代理)
    params_grads_inner_prod = None      # 初始化累积原始参数和 LoRA 更新内积的和为 None

    # 迭代参数名称、参数张量和变换类型
    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        for lora_strs in LORA_NAMES:
            if lora_strs[0] in p_name:
                lora_A_name = p_name.replace(lora_strs[0], lora_strs[1])
                lora_A = global_params[lora_A_name]
                lora_BA = torch.matmul(param, lora_A)
                original_param_name = p_name.split(lora_strs[0])[0] + 'weight'
                original_param = global_params[original_param_name]

            param_transform = None      # 对原始参数进行变换
            # 如果变换类型是多头注意力头维度
            if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                # 使用指定的分组数和头数进行变换
                param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'], param_group['num_heads'])
            elif lora_strs[0] == 'lora_embedding_B':
                param_transform = tensor_transformation(original_param, TensorTransform.TRANSPOSE, param_group['num_groups'])
            else:
                # 使用指定的分组数进行变换
                param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'])

            # 累积计算变换后原始参数的 L2 范数的平方和
            if norm_params == None: # 使用 is None
                # 如果是第一个，直接赋值
                norm_params = torch.norm(param_transform, dim=1) ** 2
            else:
                # 否则，累加
                norm_params += torch.norm(param_transform, dim=1) ** 2

            # 对 LoRA 更新 (lora_BA) 进行与原始参数相同的变换
            grad_transform = None
            # 如果变换类型是多头注意力头维度
            if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                # 使用指定的分组数和头数进行变换
                grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'], param_group['num_heads'])
            else:
                # 使用指定的分组数进行变换
                grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'])

            # 累积计算变换后 LoRA 更新的 L2 范数的平方和 (将其视为梯度的代理)
            if norm_grads == None: # 使用 is None
                # 如果是第一个，直接赋值
                norm_grads = torch.norm(grad_transform, dim=1) ** 2
            else:
                # 否则，累加
                norm_grads += torch.norm(grad_transform, dim=1) ** 2

            # 累积计算变换后原始参数和 LoRA 更新的内积
            if params_grads_inner_prod == None: # 使用 is None
                # 如果是第一个，直接赋值
                params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
            else:
                # 否则，累加
                params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)

    # 计算累积范数平方和的平方根，得到最终的 L2 范数
    norm_params = torch.sqrt(norm_params)
    norm_grads = torch.sqrt(norm_grads)
    # 计算余弦相似度：内积 / (原始参数范数 * LoRA 更新范数)
    # 添加 1e-8 防止除零，+1 将范围映射到 [0, 2]
    cosine_sim = params_grads_inner_prod / (norm_params + 1e-8) / (norm_grads + 1e-8)
    # 取绝对值，并移除 +1
    param_group['importance_scores']['cosine_similarity'] = torch.abs(cosine_sim)
'''

import torch
import torch.nn.functional as F
from only_train_once.transform import tensor_transformation, TensorTransform

LORA_NAMES = [('lora_B', 'lora_A'), ('lora_embedding_B', 'lora_embedding_A')]

# ==============================================================================
# 修改版：基于趋势的余弦相似度 (Movement Pruning Style)
# 逻辑：区分“参数生长”与“参数衰减”。
# 范围：映射到 [0, 1]，其中 1.0 代表强烈生长（最重要），0.0 代表强烈衰减（最冗余）。
# ==============================================================================

def importance_score_by_cosine_similarity(param_group):
    """
    计算基于梯度的余弦相似度分数 (Movement Pruning 策略)。
    
    [修改说明]: 
    移除了 torch.abs()。采用了 (1 - cos) / 2 的映射。
    - 梯度与参数反向 (cos=-1) -> Update会让参数变大 -> Score = 1.0 (保留)
    - 梯度与参数同向 (cos= 1) -> Update会让参数变小 -> Score = 0.0 (剪枝)
    """
    norm_params = None                  
    norm_grads = None                   
    params_grads_inner_prod = None      

    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        if p_name not in param_group['grad_variant']:
            continue
        grad = param_group['grad_variant'][p_name]

        # --- 1. 参数变换 ---
        param_transform = None      
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            param_transform = tensor_transformation(param, p_transform, param_group['num_groups'])

        if norm_params is None:
            norm_params = torch.norm(param_transform, dim=1) ** 2
        else:
            norm_params += torch.norm(param_transform, dim=1) ** 2

        # --- 2. 梯度变换 ---
        grad_transform = None       
        if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
            grad_transform = tensor_transformation(grad, p_transform, param_group['num_groups'], param_group['num_heads'])
        else:
            grad_transform = tensor_transformation(grad, p_transform, param_group['num_groups'])

        if norm_grads is None:
            norm_grads = torch.norm(param_transform, dim=1) ** 2 # 注意：此处通常应用 grad_transform 的范数，原代码可能有笔误，建议检查
            # 修正建议：应该是 norm_grads += torch.norm(grad_transform, dim=1) ** 2
            # 这里保持和你提供的一致，或者修正为 grad_transform：
            norm_grads = torch.norm(grad_transform, dim=1) ** 2 
        else:
            norm_grads += torch.norm(grad_transform, dim=1) ** 2

        # --- 3. 内积 ---
        if params_grads_inner_prod is None:
            params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
        else:
            params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)

    if norm_params is None or norm_grads is None or params_grads_inner_prod is None:
        ref_param = param_group['params'][0]
        param_group['importance_scores']['cosine_similarity'] = torch.zeros(
            param_group['num_groups'],
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
        return

    norm_params = torch.sqrt(norm_params)
    norm_grads = torch.sqrt(norm_grads)

    # 计算原始余弦值 [-1, 1]
    cosine_sim = params_grads_inner_prod / (norm_params + 1e-8) / (norm_grads + 1e-8)

    # [核心修改] Movement Pruning 映射
    # 梯度下降中，更新量是 -grad。
    # 如果 cos(w, g) = -1 (反向)，则 -grad 与 w 同向 -> 生长 -> 重要性高(1.0)
    # 如果 cos(w, g) =  1 (同向)，则 -grad 与 w 反向 -> 衰减 -> 重要性低(0.0)
    score = (1.0 - cosine_sim) / 2.0

    param_group['importance_scores']['cosine_similarity'] = score


def importance_score_by_cosine_similarity_lora(param_group, global_params):
    """
    计算 LoRA 参数的余弦相似度分数。
    
    [修改说明]: 
    移除了 torch.abs()。采用了 (1 + cos) / 2 的映射。
    - LoRA更新量 BA 与 原参数 W 同向 (cos= 1) -> 强化参数 -> Score = 1.0 (保留)
    - LoRA更新量 BA 与 原参数 W 反向 (cos=-1) -> 抵消参数 -> Score = 0.0 (剪枝)
    注意：这里与梯度版公式相反，因为 LoRA 是直接相加 (W + BA)，而梯度是相减 (W - \eta G)。
    """
    norm_params = None                  
    norm_grads = None                   
    params_grads_inner_prod = None      

    for p_name, param, p_transform in zip(param_group['p_names'], param_group['params'], param_group['p_transform']):
        for lora_strs in LORA_NAMES:
            if lora_strs[0] in p_name:
                lora_A_name = p_name.replace(lora_strs[0], lora_strs[1])
                lora_A = global_params[lora_A_name]
                # LoRA update term BA
                lora_BA = torch.matmul(param, lora_A) 
                original_param_name = p_name.split(lora_strs[0])[0] + 'weight'
                original_param = global_params[original_param_name]

            # --- 1. 原始参数变换 ---
            param_transform = None      
            if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'], param_group['num_heads'])
            elif lora_strs[0] == 'lora_embedding_B':
                param_transform = tensor_transformation(original_param, TensorTransform.TRANSPOSE, param_group['num_groups'])
            else:
                param_transform = tensor_transformation(original_param, p_transform, param_group['num_groups'])

            if norm_params is None:
                norm_params = torch.norm(param_transform, dim=1) ** 2
            else:
                norm_params += torch.norm(param_transform, dim=1) ** 2

            # --- 2. LoRA Update (BA) 变换 (作为梯度的代理/更新向量) ---
            grad_transform = None
            if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'], param_group['num_heads'])
            else:
                grad_transform = tensor_transformation(lora_BA, p_transform, param_group['num_groups'])

            if norm_grads is None:
                norm_grads = torch.norm(grad_transform, dim=1) ** 2
            else:
                norm_grads += torch.norm(grad_transform, dim=1) ** 2

            # --- 3. 内积 ---
            if params_grads_inner_prod is None:
                params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
            else:
                params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)

    if norm_params is None or norm_grads is None or params_grads_inner_prod is None:
        ref_param = param_group['params'][0]
        param_group['importance_scores']['cosine_similarity'] = torch.zeros(
            param_group['num_groups'],
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
        return

    norm_params = torch.sqrt(norm_params)
    norm_grads = torch.sqrt(norm_grads)
    
    cosine_sim = params_grads_inner_prod / (norm_params + 1e-8) / (norm_grads + 1e-8)
    
    # [核心修改] LoRA Movement Logic
    # 最终权重 = W_orig + BA
    # 如果 cos(W, BA) = 1 (同向)，则权重增强 -> 生长 -> 重要性高(1.0)
    # 如果 cos(W, BA) = -1 (反向)，则权重被抵消 -> 衰减 -> 重要性低(0.0)
    score = (1.0 + cosine_sim) / 2.0
    
    param_group['importance_scores']['cosine_similarity'] = score

