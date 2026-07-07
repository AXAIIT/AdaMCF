from abc import ABC, abstractclassmethod  # 导入抽象基类和抽象类方法装饰器
from torch.optim.optimizer import Optimizer, required 
import torch 
from .base_optimizer import BaseOptimizer  # 导入自定义的基础优化器类
from .hyperparameter import DEFAULT_OPT_PARAMS, SUPPORT_GRADIENT_ESTIMATES  # 导入预定义的优化器参数和支持的梯度估计方法
from only_train_once.transform import tensor_transformation, TensorTransform, index_transformation, index_transformation_param_group  # 导入张量变换相关函数
from .importance_score import calculate_importance_score  # 导入计算重要性分数的函数
import numpy as np  

class SparseOptimizerMetrics:
    """
    稀疏优化器的指标跟踪类，用于记录和存储优化过程中的各种统计指标
    """
    num_groups = 0  # 总组数
    num_zero_groups = 0  # 零权重组的数量
    num_important_groups = 0  # 重要组的数量
    num_redundant_groups = 0  # 冗余组的数量
    num_pruned_groups = 0  # 被剪枝的组数量
    num_active_redundant_groups = 0  # 仍然活跃的冗余组数量
    num_unpruned_groups = 0  # 未被剪枝的组数量
    
    # 用于CRIC (Conflicting Rules in Implementation Checking)
    num_violating_groups = 0  # 违反约束的组的数量
    num_trial_violating_groups = 0  # 试验中违反约束的组的数量
    num_historical_violating_groups = 0  # 历史上违反约束的组的数量
    
    norm_violating_groups = 0.0  # 违反约束的组的范数总和

    norm_params = 0.0  # 所有参数的范数总和
    norm_important_groups = 0.0  # 重要组的范数总和
    norm_redundant_groups = 0.0  # 冗余组的范数总和

    group_sparsity = 0.0  # 组稀疏度（零组比例）
    zero_group_ratio = 0.0  # 零权重组所占比例
    pruned_group_ratio = 0.0  # 剪枝组比例
    redundant_group_ratio = 0.0  # 冗余组比例
    important_group_ratio = 0.0  # 重要组比例
    active_redundant_group_ratio = 0.0  # 活跃冗余组比例

    flops_before = 0.0  # 剪枝前 FLOPs
    flops_after = 0.0  # 剪枝后 FLOPs
    pruned_flops = 0.0  # 剪枝减少的 FLOPs
    pruned_flops_percent = 0.0  # 剪枝减少的 FLOPs 百分比
    flops_compression_ratio = 0.0  # FLOPs 压缩比

    params_before = 0.0  # 剪枝前参数量
    params_after = 0.0  # 剪枝后参数量
    pruned_params = 0.0  # 减少的参数量
    pruned_params_percent = 0.0  # 减少的参数量百分比
    params_compression_ratio = 0.0  # 参数量压缩比

class BaseHybridSparseOptimizer(BaseOptimizer):
    """
    混合稀疏优化器的基类，继承自BaseOptimizer
    实现了对模型参数进行组级别稀疏化的功能
    """
    def __init__(self, params, variant='sgd', lr=required, \
                 first_momentum=None, second_momentum=None, dampening=None, weight_decay=None, \
                 target_group_sparsity=0.0, group_divisible=1, additional_defaults=dict()):
        """
        初始化混合稀疏优化器
        
        参数:
            params: 待优化参数的迭代器，包含参数和相关元数据
            variant: 优化算法变体，如'sgd'或'adam'
            lr: 学习率
            first_momentum: 一阶动量系数
            second_momentum: 二阶动量系数
            dampening: 阻尼系数
            weight_decay: 权重衰减系数
            target_group_sparsity: 目标组稀疏度（要剪枝的组的比例）
            group_divisible: 组大小的除数，用于确定可剪枝的最小组
            additional_defaults: 额外的默认参数字典
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if variant not in SUPPORT_GRADIENT_ESTIMATES:
            raise ValueError("Need to select a gradient estimation from {}".format(SUPPORT_GRADIENT_ESTIMATES))
        
        # 设置与基础优化器相关的超参数
        first_momentum = first_momentum if first_momentum is not None else DEFAULT_OPT_PARAMS[variant]['first_momentum']
        second_momentum = second_momentum if second_momentum is not None else DEFAULT_OPT_PARAMS[variant]['second_momentum']
        dampening = dampening if dampening is not None else DEFAULT_OPT_PARAMS[variant]['dampening']
        weight_decay = weight_decay if weight_decay is not None else DEFAULT_OPT_PARAMS[variant]['weight_decay']

        # 设置默认参数
        defaults = dict(lr=lr, weight_decay=weight_decay, first_momentum=first_momentum, second_momentum=second_momentum, \
                        dampening=dampening, variant=variant, grad_variant=dict(), global_start_idx=0, global_idx=0)
        defaults.update(additional_defaults)  # 合并额外默认参数

        super(BaseHybridSparseOptimizer, self).__init__(params, defaults)  # 调用父类初始化方法

        # 设置可剪枝组的总数
        self.total_num_groups = 0
        for param_group in params:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:  # 如果参数组可以剪枝且不是辅助参数
                if param_group['num_groups'] <= group_divisible:  # 如果组数小于等于除数，则设为不可剪枝
                    param_group['is_prunable'] = False
                else:
                    self.total_num_groups += param_group['num_groups']  # 累计可剪枝的组数

        self.group_divisible = group_divisible  # 组大小除数
        self.target_group_sparsity = target_group_sparsity  # 目标组稀疏度
        # 计算目标冗余组数（要剪枝的组数）
        self.target_num_redundant_groups = int(self.total_num_groups * min(self.target_group_sparsity, 0.999))
        self.opt_metrics = SparseOptimizerMetrics()  # 初始化指标跟踪器

        # 将辅助参数组存储在字典中，以便快速访问
        self.auxiliary_param_groups = dict()
        for group in self.param_groups:
            if group['is_auxiliary']:
                self.auxiliary_param_groups[group['id']] = group
        
    def gradient_descent_step(self, param_group):
        """
        执行梯度下降步骤，更新参数
        
        参数:
            param_group: 要更新的参数组
        """
        for p_name, p in zip(param_group['p_names'], param_group['params']):
            if p_name not in param_group['grad_variant']:  # 如果没有对应的梯度变体，则跳过
                continue
            # 如果是AdamW，则单独处理权重衰减
            if param_group['weight_decay'] is not None and param_group['variant'] == 'adamw':
                p.data.add_(param_group['weight_decay'] * p.data, alpha=-param_group['lr'])
            # 应用梯度更新
            p.data.add_(param_group['grad_variant'][p_name], alpha=-param_group['lr'])

    def fix_pruned_groups_as_zeros(self, param_group):
        """
        确保已剪枝的组参数保持为零
        
        参数:
            param_group: 包含已剪枝组的参数组
        """
        if len(param_group['pruned_idxes']) > 0:  # 如果有被剪枝的索引
            for p, p_transform in zip(param_group['params'], param_group['p_transform']):
                # 转换剪枝索引以匹配参数的变换方式
                pruned_idxes = index_transformation_param_group(param_group['pruned_idxes'], p_transform, param_group)
                # 根据变换类型处理参数
                if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                    p.data[:, pruned_idxes] = 0.0  # 对转置的参数设零
                else:
                    p.data[pruned_idxes] = 0.0  # 对普通参数设零
                    
            # 处理相关的辅助参数
            for ng_id, offset in param_group['auxiliary_ngs']:
                pruned_aux_idxes = [i + offset for i in pruned_idxes]  # 计算辅助参数中对应的索引
                for aux_p in self.auxiliary_param_groups[ng_id]['params']:
                    if aux_p.grad is None:
                        continue

                    # 添加边界检查，筛选有效索引
                    valid_idxes = [i for i in pruned_aux_idxes if 0 <= i < aux_p.data.size(0)]
                    if not valid_idxes:
                        # 如果没有有效索引，跳过当前辅助参数
                        continue

                    aux_p.data[valid_idxes, ...] = 0.0

                    # aux_p.data[pruned_aux_idxes, ...] = 0.0  # 将辅助参数对应位置设置为零

    def compute_importance_scores(self, **kwargs):
            """
            计算所有参数组的重要性分数。
            这个函数是剪枝决策的核心，它分三步执行：
            1. 为每个可剪枝的参数组计算各种原始的重要性分数（如幅度、泰勒展开等）。
            2. 对每种重要性标准的所有分数进行全局L2规范化，以消除不同标准之间的量纲差异。
            3. 将规范化后的分数根据用户定义的权重进行加权求和，得到一个最终的“总体”重要性分数。
            
            参数:
                **kwargs: 额外的关键字参数，为未来扩展保留。
            """
            # 初始化全局起始索引，用于后续为每个组的重要性分数分配一个全局唯一的索引范围。
            global_start_idx = 0
            # 初始化一个列表，用于收集所有参数组的最终“总体”重要性分数，以便进行全局排序。
            self.global_scores = list()
            
            # --- 步骤 1: 计算原始重要性分数 ---
            # 遍历优化器中的所有参数组。
            for group in self.param_groups:
                # 只对被标记为“可剪枝”且非“辅助”的参数组进行操作。
                if group['is_prunable'] and not group['is_auxiliary']:
                    # 调用一个统一的接口函数，该函数会根据 self.importance_score_criteria 的配置，
                    # 依次调用所有指定的重要性评分标准的计算函数（例如，计算幅度、Fisher信息等）。
                    # 计算结果会直接存储在 group['importance_scores'] 字典中。
                    calculate_importance_score(self.importance_score_criteria, group)

            # --- 步骤 2: 对每种重要性标准进行全局规范化 ---
            # 计算用于规范化的分母。
            # 首先，为每个重要性标准（例如 'magnitude', 'taylor_first_order'）初始化一个分母，其初始值为一个很小的安全数，以防除以零。
            normalization_denoms = dict.fromkeys(self.importance_score_criteria.keys(), self.safe_guard)
            # 再次遍历所有参数组。
            for group in self.param_groups:
                # 同样，只处理可剪枝且非辅助的组。
                if group['is_prunable'] and not group['is_auxiliary']:
                    # 遍历用户配置中指定的每一种重要性标准。
                    for proxy_name in self.importance_score_criteria:
                        # 如果当前组由于某种原因没有计算出该标准的分数，则跳过。
                        if not proxy_name in group['importance_scores']:
                            continue
                        # 核心步骤：计算该标准在所有组上的“总能量”（L2范数的平方）。
                        # 1. group['importance_scores'][proxy_name] ** 2：将当前组的该项分数向量中的每个元素平方。
                        # 2. torch.sum(...)：将平方后的所有元素求和。
                        # 3. .item()：将结果从tensor转换为Python浮点数。
                        # 4. normalization_denoms[proxy_name] += ...：将当前组的能量累加到该标准的全局总能量上。
                        normalization_denoms[proxy_name] += torch.sum(group['importance_scores'][proxy_name] ** 2, dim=0).item()
            
            # 对每个标准累加的平方和取平方根，得到最终的全局L2范数作为规范化因子。
            for proxy_name in normalization_denoms:
                # 加上一个安全数以确保分母不为零。
                normalization_denoms[proxy_name] = np.sqrt(normalization_denoms[proxy_name]) + self.safe_guard

            # --- 步骤 3: 应用规范化、加权并计算总体分数 ---
            # 重置全局起始索引，用于最终的全局索引分配。
            global_start_idx = 0
            # 第三次遍历所有参数组。
            for group in self.param_groups:
                # 依然只处理可剪枝且非辅助的组。
                if group['is_prunable'] and not group['is_auxiliary']:
                    # 为当前组初始化一个空的“总体”重要性分数。
                    group['importance_scores']['overall'] = None
                    # 遍历每一种重要性标准。
                    for proxy_name in self.importance_score_criteria:
                        # 如果分数不存在则跳过。
                        if not proxy_name in group['importance_scores']:
                            continue
                        # 核心步骤：应用权重并规范化。
                        # 1. self.importance_score_criteria[proxy_name]：获取用户为该标准设定的权重。
                        # 2. ... / normalization_denoms[proxy_name]：除以该标准全局的L2范数进行规范化。
                        # 3. .mul_(...)：将计算出的加权规范化因子原地乘到原始分数上。
                        group['importance_scores'][proxy_name].mul_(self.importance_score_criteria[proxy_name] / normalization_denoms[proxy_name])
                        
                        # 将处理后的分数累加到“总体”分数上。
                        if group['importance_scores']['overall'] is None:
                            # 如果是第一个标准，直接克隆其分数作为总体分数的初始值。
                            group['importance_scores']['overall'] = group['importance_scores'][proxy_name].clone()
                        else:
                            # 对于后续的标准，将其分数累加到总体分数上。
                            group['importance_scores']['overall'] += group['importance_scores'][proxy_name]
                    
                    # 为该组记录其在全局分数列表中的索引信息。
                    group['global_start_idx'] = global_start_idx
                    group['global_idxes'] = np.arange(global_start_idx, global_start_idx+group['num_groups'])
                    # 更新全局起始索引，为下一个组做准备。
                    global_start_idx += group['num_groups']
                    # 将当前组计算出的最终“总体”重要性分数添加到全局分数列表中。
                    self.global_scores.append(group['importance_scores']['overall'])

    def compute_metrics(self, **summary_stats):
        """
        计算并返回优化过程中的各种指标
        
        返回:
            SparseOptimizerMetrics: 包含各种稀疏化指标的对象
        """
        # 初始化指标
        self.opt_metrics.norm_params = 0.0
        self.opt_metrics.norm_important_groups = 0.0
        self.opt_metrics.norm_redundant_groups = 0.0
        self.opt_metrics.num_zero_groups = 0
        self.opt_metrics.num_important_groups = 0
        self.opt_metrics.num_redundant_groups = 0
        self.opt_metrics.num_pruned_groups = 0
        self.opt_metrics.num_active_redundant_groups = 0
        self.opt_metrics.num_unpruned_groups = 0
        self.opt_metrics.group_sparsity = 0.0
        self.opt_metrics.zero_group_ratio = 0.0
        self.opt_metrics.pruned_group_ratio = 0.0
        self.opt_metrics.redundant_group_ratio = 0.0
        self.opt_metrics.important_group_ratio = 0.0
        self.opt_metrics.active_redundant_group_ratio = 0.0
        self.opt_metrics.flops_before = 0.0
        self.opt_metrics.flops_after = 0.0
        self.opt_metrics.pruned_flops = 0.0
        self.opt_metrics.pruned_flops_percent = 0.0
        self.opt_metrics.flops_compression_ratio = 0.0
        self.opt_metrics.params_before = 0.0
        self.opt_metrics.params_after = 0.0
        self.opt_metrics.pruned_params = 0.0
        self.opt_metrics.pruned_params_percent = 0.0
        self.opt_metrics.params_compression_ratio = 0.0

        total_groups = 0
        total_pruned_groups = 0
        total_active_redundant_groups = 0

        for group in self.param_groups:
            if not (group['is_prunable'] and not group['is_auxiliary']):  # 跳过不可剪枝或辅助参数组
                continue
            total_groups += group.get('num_groups', 0)
            norm_group = None  # 初始化组范数
            import_idxes = group.get('important_idxes', [])  # 获取重要组的索引
            redund_idxes = group.get('active_redundant_idxes', []) + group.get('pruned_idxes', [])  # 获取冗余组的索引
            total_pruned_groups += len(group.get('pruned_idxes', []))
            total_active_redundant_groups += len(group.get('active_redundant_idxes', []))

            # 计算每个参数的组范数
            for param, p_transform in zip(group['params'], group['p_transform']):
                if p_transform == TensorTransform.NO_PRUNE:  # 跳过不剪枝的变换
                    continue
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(param.data, p_transform, group['num_groups'], group['num_heads'])
                else:
                    param_transform = tensor_transformation(param.data, p_transform, group['num_groups'])
                if norm_group is None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2

            if norm_group is None:
                continue

            norm_group = torch.sqrt(norm_group)  # 计算平方根得到最终范数
            
            zero_groups = int(torch.sum(norm_group == 0).item())
            self.opt_metrics.num_zero_groups += zero_groups  # 零组数量
            self.opt_metrics.norm_params += torch.sum(norm_group).item()  # 参数范数总和
            self.opt_metrics.norm_important_groups += torch.sum(norm_group[import_idxes]).item() if len(import_idxes) > 0 else 0.0  # 重要组范数总和
            self.opt_metrics.norm_redundant_groups += torch.sum(norm_group[redund_idxes]).item() if len(redund_idxes) > 0 else 0.0  # 冗余组范数总和
            self.opt_metrics.num_important_groups += len(import_idxes)  # 重要组数量
            self.opt_metrics.num_redundant_groups += len(redund_idxes)  # 冗余组数量

        safe_total_groups = float(total_groups + self.safe_guard) if total_groups > 0 else float(self.safe_guard if self.safe_guard > 0 else 1.0)
        denom = float(total_groups) if total_groups > 0 else 1.0

        self.opt_metrics.num_groups = total_groups
        self.opt_metrics.num_pruned_groups = total_pruned_groups
        self.opt_metrics.num_active_redundant_groups = total_active_redundant_groups
        self.opt_metrics.num_unpruned_groups = max(total_groups - total_pruned_groups, 0)

        self.opt_metrics.group_sparsity = self.opt_metrics.num_zero_groups / safe_total_groups
        self.opt_metrics.zero_group_ratio = self.opt_metrics.num_zero_groups / denom if total_groups > 0 else 0.0
        self.opt_metrics.pruned_group_ratio = total_pruned_groups / denom if total_groups > 0 else 0.0
        self.opt_metrics.redundant_group_ratio = self.opt_metrics.num_redundant_groups / denom if total_groups > 0 else 0.0
        self.opt_metrics.important_group_ratio = self.opt_metrics.num_important_groups / denom if total_groups > 0 else 0.0
        self.opt_metrics.active_redundant_group_ratio = total_active_redundant_groups / denom if total_groups > 0 else 0.0

        flops_before = summary_stats.get('flops_before')
        flops_after = summary_stats.get('flops_after')
        if isinstance(flops_before, (int, float)) and isinstance(flops_after, (int, float)):
            pruned_flops = max(flops_before - flops_after, 0.0)
            self.opt_metrics.flops_before = float(flops_before)
            self.opt_metrics.flops_after = float(flops_after)
            self.opt_metrics.pruned_flops = pruned_flops
            self.opt_metrics.pruned_flops_percent = (pruned_flops / flops_before * 100.0) if flops_before > 0 else 0.0
            self.opt_metrics.flops_compression_ratio = (flops_before / flops_after) if flops_after > 0 else float('inf')
        else:
            self.opt_metrics.flops_before = float(flops_before) if isinstance(flops_before, (int, float)) else 0.0
            self.opt_metrics.flops_after = float(flops_after) if isinstance(flops_after, (int, float)) else 0.0
            self.opt_metrics.pruned_flops = 0.0
            self.opt_metrics.pruned_flops_percent = 0.0
            self.opt_metrics.flops_compression_ratio = 0.0

        params_before = summary_stats.get('params_before')
        params_after = summary_stats.get('params_after')
        if isinstance(params_before, (int, float)) and isinstance(params_after, (int, float)):
            pruned_params = max(params_before - params_after, 0.0)
            self.opt_metrics.params_before = float(params_before)
            self.opt_metrics.params_after = float(params_after)
            self.opt_metrics.pruned_params = pruned_params
            self.opt_metrics.pruned_params_percent = (pruned_params / params_before * 100.0) if params_before > 0 else 0.0
            self.opt_metrics.params_compression_ratio = (params_before / params_after) if params_after > 0 else float('inf')
        else:
            self.opt_metrics.params_before = float(params_before) if isinstance(params_before, (int, float)) else 0.0
            self.opt_metrics.params_after = float(params_after) if isinstance(params_after, (int, float)) else 0.0
            self.opt_metrics.pruned_params = 0.0
            self.opt_metrics.pruned_params_percent = 0.0
            self.opt_metrics.params_compression_ratio = 0.0

        return self.opt_metrics
