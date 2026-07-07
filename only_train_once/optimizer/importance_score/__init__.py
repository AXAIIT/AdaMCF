'''
(__init__.py)主要作用是：
聚合重要性评分方法：它从同目录下的 magnitude.py, cosine_similarity.py, 和 taylor.py 模块中导入各种计算参数重要性分数的功能。
提供统一计算接口：定义了两个核心函数 calculate_importance_score_dhspg 和 calculate_importance_score_lhspg。
这两个函数根据传入的 criteria (标准列表) 来选择并调用相应的具体重要性评分计算函数（例如，基于幅值、余弦相似度、泰勒展开等）。
区分不同策略：函数名中的 dhspg 和 lhspg 可能代表两种不同的参数分组或剪枝策略，这两个函数分别为这两种策略计算重要性分数，
并将结果存储在 param_group 字典的 'importance_scores' 键中。
该文件是计算模型参数重要性分数的入口点，整合了多种评分标准，并为不同的优化或剪枝上下文提供了计算接口。
'''

from .L1 import *                       # 从同级目录的 L1 模块导入所有内容 (包含基于 L1 范数的重要性计算函数)
from .L2 import *                       # 从同级目录的 L2 模块导入所有内容 (包含基于幅值的重要性计算函数)
from .cosine_similarity import *        # 从同级目录的 cosine_similarity 模块导入所有内容 (包含基于余弦相似度的重要性计算函数)
from .taylor import *                   # 从同级目录的 taylor 模块导入所有内容 (包含基于泰勒展开的重要性计算函数)
from .fisher import *                   # 从同级目录的 fisher 模块导入所有内容 (包含基于 Fisher 信息的重要性计算函数)
from .grad_weight_product import *       # 从同级目录的 grad_weight_product 模块导入所有内容 (包含基于梯度和权重乘积的重要性计算函数)
import torch                            # 导入 PyTorch 库


def calculate_importance_score(criteria, param_group, criteria_config=None):
    """
    计算参数重要性分数。
    Args:
        criteria (list): 一个包含重要性评分标准名称的列表 (例如 ['magnitude', 'taylor_first_order'])。
        param_group (dict): 优化器中的参数组字典，包含参数张量等信息。此函数会向该字典添加 'importance_scores' 键。
    """
    param_group['importance_scores'] = dict()       # 在参数组字典中初始化一个用于存储重要性分数的空字典
    # 使用 torch.no_grad() 上下文管理器，确保在此块中的计算不被追踪梯度，以节省内存和计算资源
    with torch.no_grad():
        # 遍历用户指定的每个重要性评分标准
        for cri_name in criteria:
            # 根据标准名称调用相应的计算函数
            if 'l2_magnitude' == cri_name:
                importance_score_by_l2_magnitude(param_group)                # 计算基于 L2 范数参数幅值的重要性分数
            elif 'l2_avg_magnitude' == cri_name:
                importance_score_by_l2_avg_magnitude(param_group)            # 计算基于 L2 范数参数平均幅值的重要性分数
            elif 'l1_magnitude' in cri_name:
                importance_score_by_l1_magnitude(param_group)               # 计算基于 L1 范数的重要性分数
            elif 'l1_avg_magnitude' in cri_name:
                importance_score_by_l1_avg_magnitude(param_group)           # 计算基于 L1 范数参数平均幅值的重要性分数
            elif 'cosine_similarity' == cri_name:
                importance_score_by_cosine_similarity(param_group)          # 计算基于余弦相似度的重要性分数
            elif 'taylor_first_order' == cri_name:
                importance_score_by_first_order_taylor(param_group)         # 计算基于一阶泰勒展开的重要性分数
            elif 'taylor_second_order' == cri_name:
                importance_score_by_second_order_taylor(param_group)        # 计算基于二阶泰勒展开的重要性分数
            elif 'fisher' in cri_name:
                importance_score_by_fisher(param_group)                     # 计算基于 fisher 的重要性分数
            elif 'grad_weight_product' in cri_name:
                importance_score_by_grad_weight_product(param_group)        # 计算基于 grad_weight_product 的重要性分数
            

# 定义一个函数，用于计算 'lora' 策略下的参数重要性分数
def calculate_importance_score_lora(criteria, param_group, global_params, criteria_config=None):
    """
    计算参数重要性分数。
    Args:
        criteria (list): 一个包含重要性评分标准名称的列表。
        param_group (dict): 优化器中的参数组字典。此函数会向该字典添加 'importance_scores' 键（如果尚不存在）。
        global_params (list): 可能包含模型全局参数或其他相关信息的列表，某些评分标准需要用到。
    """
    # 同样在无梯度追踪的上下文中执行计算
    with torch.no_grad():
        # 遍历用户指定的每个重要性评分标准
        for cri_name in criteria:
            # 根据标准名称调用相应的计算函数
            if 'l2_magnitude' in cri_name:
                importance_score_by_l2_magnitude_lora(param_group)                          # 计算基于 L2 范数参数幅值的重要性分数
            elif 'l2_avg_magnitude' == cri_name:
                importance_score_by_l2_avg_magnitude_lora(param_group)                      # 计算基于 L2 范数参数平均幅值的重要性分数
            elif 'l1_magnitude' in cri_name:
                importance_score_by_l1_magnitude_lora(param_group)                          # 计算基于 L1 范数的重要性分数，排除 LoRA 参数
            elif 'l1_avg_magnitude' in cri_name:
                importance_score_by_l1_avg_magnitude_lora(param_group)                      # 计算基于 L1 范数的平均重要性分数，排除 LoRA 参数
            elif 'cosine_similarity' in cri_name:
                importance_score_by_cosine_similarity_lora(param_group, global_params)      # 计算基于余弦相似度的重要性分数，需要额外的 global_params
            elif 'taylor_first_order' in cri_name:
                importance_score_by_first_order_taylor_lora(param_group, global_params)     # 计算基于一阶泰勒展开的重要性分数，需要额外的 global_params
            elif 'taylor_second_order' in cri_name:
                importance_score_by_second_order_taylor_lora(param_group, global_params)    # 计算基于二阶泰勒展开的重要性分数，需要额外的 global_params
            elif 'fisher' in cri_name:
                importance_score_by_fisher_lora(param_group)                                # 计算基于 fisher 范数的平均重要性分数，排除 LoRA 参数
            elif 'grad_weight_product' in cri_name:
                importance_score_by_grad_weight_product_lora(param_group)                   # 计算基于 grad_weight_product 范数的平均重要性分数，排除 LoRA 参数
