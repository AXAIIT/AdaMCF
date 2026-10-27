import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
import torch.nn.functional as F
from only_train_once.transform import TensorTransform, index_transformation_param_group, pruning_axis
from .base_hybrid_sparse_optimizer import BaseHybridSparseOptimizer
from only_train_once.graph import Graph
import random


class HESSO(BaseHybridSparseOptimizer):
    """
    HESSO: Hybrid Efficient Structured Sparse Optimizer (混合高效结构化稀疏优化器)
    功能:
    HESSO 是一种用于实现神经网络结构化稀疏性的优化器。它旨在通过以下方式工作：
    1. 评估参数组的重要性：利用二阶信息（Hessian 的近似或代理）以及其他可能的标准（如梯度、参数大小）来计算模型中参数组（例如，卷积层中的整个滤波器或通道，或 Transformer 中的注意力头）的重要性分数。
    2. 逐步稀疏化：根据计算出的重要性分数，在训练过程中逐步识别并剪枝（置零）那些不重要的参数组。
    3. 结构化剪枝：与非结构化剪枝（剪枝单个权重）不同，HESSO 移除整个参数组，这通常能带来实际的计算加速和模型大小缩减。
    4. 混合策略：名称中的 "Hybrid" 可能暗示它结合了多种技术或标准来评估重要性或执行剪枝。
    参数 (在 __init__ 中定义):
    - params: 模型的参数组，通常由 Graph 类提供，包含了每个组的详细信息（如是否可剪枝、是否为辅助组、组内参数等）。
    - variant: 基础优化算法的类型（如 'sgd', 'adam'），HESSO 在此基础上添加剪枝逻辑。
    - lr, weight_decay, momentums: 标准的优化器超参数。
    - target_group_sparsity: 目标要达到的参数组稀疏度（被剪掉的组的比例）。
    - start_pruning_step, pruning_steps, pruning_periods: 控制剪枝过程的时间表和频率。
    - group_divisible: 剪枝后剩余组数量需要满足的可除性约束。
    - fixed_zero_groups: 是否在剪枝后固定已置零的组，防止它们被重新更新。
    - importance_score_criteria: 用于计算参数组重要性的不同标准的权重。
    - device: 设备类型（如 'cuda' 或 'cpu'），用于存储和计算。
    - tolerance_group_sparsity: 容忍的组稀疏度范围，用于判断是否达到目标稀疏度。
    - first_momentum: 一阶动量参数，控制梯度更新的平滑程度。
    - second_momentum: 二阶动量参数，通常用于 Adam 优化器。
    - dampening: 动量的抑制因子，控制动量的衰减。
    核心步骤 (在 step 方法中体现):
    1. 计算梯度变体：根据基础优化器类型（如 SGD, Adam）和动量等计算有效的梯度。
    2. 识别冗余组：在剪枝周期内，如果未达到目标稀疏度，则：
        a. 计算重要性分数：对可剪枝的参数组，根据预设标准计算其重要性。
        b. 确定冗余组：根据重要性分数，选出当前周期要剪枝的冗余组。
    3. 更新参数：
        a. 对于非冗余组或不可剪枝组，正常更新参数。
        b. 对于当前周期被标记为冗余的组，其参数在更新后会逐渐衰减至零。
        c. 对于已经完全剪枝的组，其参数保持为零。
    4. 提交冗余组：在剪枝周期结束时，将当前周期识别的冗余组正式标记为已剪枝。
    """

    @staticmethod
    def _build_period_quota(total_groups: int, periods: int, mode: str = "cosine_front"):
        import numpy as np
        if periods <= 1 or total_groups <= 0:
            return [max(total_groups, 0)]
        if mode == "uniform":
            # Deterministic balanced integer allocation.  Cumulative flooring
            # spreads the remainder across the horizon instead of creating an
            # artificial spike in the final period.
            return [
                ((i + 1) * total_groups) // periods
                - (i * total_groups) // periods
                for i in range(periods)
            ]
        if mode == "random":
            base = [total_groups // periods] * periods
            remainder = total_groups % periods
            indices = list(range(periods))
            random.shuffle(indices)
            for i in range(remainder):
                base[indices[i]] += 1
            return base
        if mode == "cosine_front":
            x = np.linspace(0, 1, periods, endpoint=True)
            w = 1 - np.cos(np.pi * (1 - x))  # 前期权重更大
            w = w / w.sum()
            alloc = np.floor(w * total_groups).astype(int)
            diff = total_groups - int(alloc.sum())
            if diff > 0:
                order = np.argsort(-w)
                for i in range(diff):
                    alloc[order[i % periods]] += 1
            elif diff < 0:
                order = np.argsort(w)
                for i in range(-diff):
                    idx = order[i % periods]
                    if alloc[idx] > 0:
                        alloc[idx] -= 1
            return alloc.tolist()
        raise ValueError(f"Unknown pruning quota mode: {mode}")

    def __init__(self, params, variant='sgd', lr=required, first_momentum=None, second_momentum=None, \
                 dampening=None, weight_decay=None, target_group_sparsity=0.5, \
                 start_pruning_step=0, pruning_steps=None, pruning_periods=200, \
                 group_divisible=1, importance_score_criteria='default', device='cuda'):

        # --- 规范化输出 HESSO 初始化参数 ---
        print("\n--- HESSO Optimizer Initialization ---")
        print(f"  {'Optimizer Variant:':<30} {variant}")
        print(f"  {'Learning Rate (lr):':<30} {lr}")
        print(f"  {'Target Group Sparsity:':<30} {target_group_sparsity}")
        print(f"  {'Device:':<30} {device}")
        print(f"  {'Start Pruning Step:':<30} {start_pruning_step}")
        print(f"  {'Pruning Steps (total):':<30} {pruning_steps}")
        print(f"  {'Pruning Periods:':<30} {pruning_periods}")
        print(f"  {'Group Divisible:':<30} {group_divisible}")
        print(f"  {'Importance Score Criteria:':<30} {importance_score_criteria}")
        
        # 初始化优化器的基本参数
        self.start_pruning_step = start_pruning_step                                # 开始剪枝的步数
        self.pruning_periods = int(max(1, pruning_periods))                         # 剪枝的周期数
        self.pruning_steps = pruning_steps                                          # 总的剪枝步数
        self.pruning_period_duration = self.pruning_steps // self.pruning_periods   # 每个剪枝周期的步数
        self.curr_pruning_period = 0                                                # 当前剪枝周期
        self.device = device                                                        # 设备（如 'cuda' 或 'cpu'）

        self.pruned_group_idxes = list()    # 初始化一个空列表，用于存储所有已经被剪枝的参数组的全局索引。
        # 设置参数组重要性分数的计算标准
        if importance_score_criteria == 'default':
            # 如果用户选择 'default'，则使用预定义的权重组合多种重要性评估方法。
            self.importance_score_criteria = {'magnitude': 0.2, 'avg_magnitude': 0.2, 'cosine_similarity': 0.2, 'taylor_first_order': 0.2, 'taylor_second_order': 0.2}
        else:
            # 如果用户提供了自定义的字典，则直接使用该字典作为重要性评分标准。
            self.importance_score_criteria = importance_score_criteria

        super(HESSO, self).__init__(params=params, variant=variant, lr=lr, first_momentum=first_momentum, second_momentum=second_momentum, \
                                    dampening=dampening, weight_decay=weight_decay, target_group_sparsity=target_group_sparsity, \
                                    group_divisible=group_divisible)

        for param_group in self.param_groups:
            param_group['important_idxes'] = [i for i in range(param_group['num_groups'])]
            param_group['active_redundant_idxes'] = list()
            param_group['pruned_idxes'] = list()
            param_group['importance_scores'] = dict()
        
        self.active_num_redundant_groups = self._build_period_quota(
            total_groups=self.target_num_redundant_groups,
            periods=self.pruning_periods,
            mode="cosine_front"
        )
        print(f"Target redundant groups per period: ", self.active_num_redundant_groups)

    def identify_redundant_groups(self):
        # 确定冗余组
        global_scores = torch.cat(self.global_scores, dim=0)                                            # 将全局分数拼接成一个张量
        curr_active_num_redundant_groups = self.active_num_redundant_groups[self.curr_pruning_period]   # 当前周期的冗余组数量
        curr_K = len(self.pruned_group_idxes) + curr_active_num_redundant_groups                        # 当前需要剪枝的总组数
        _, top_indices = torch.topk(-global_scores, curr_K)                                             # 获取分数最低的组索引
        top_indices = top_indices.cpu().numpy()                                                         # 转为NumPy数组
        # 排除已剪枝的组，并获取当前周期的冗余组索引
        top_indices = np.setdiff1d(top_indices, self.pruned_group_idxes)[:curr_active_num_redundant_groups].tolist()
        self.pruned_group_idxes.extend(top_indices)  # 更新已剪枝组索引
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                # 获取当前组的全局冗余索引
                global_active_redundant_idx = np.intersect1d(top_indices, group['global_idxes'])

                # 转换为局部索引
                group['active_redundant_idxes'] = (global_active_redundant_idx - group['global_start_idx']).tolist()

                group['active_redundant_idxes'] = [i for i in group['active_redundant_idxes'] if 0 <= i < group['num_groups']]

                # 根据组的可分性 (self.group_divisible) 调整冗余组和重要组的数量。
                # group_divisible 约束确保剪枝后剩余的组数量是某个整数的倍数，以适应硬件或库的优化。
                if group['num_groups'] < self.group_divisible:
                    # 如果当前参数组内的总单元数 (group['num_groups']) 小于 group_divisible 约束，则不对此组进行剪枝，清空其活动冗余索引和已剪枝索引。
                    # 这意味着非常小的组可能不会被进一步细分剪枝。
                    group['active_redundant_idxes'].clear()
                    group['important_idxes'] = [
                        i for i in range(group['num_groups'])
                        if i not in group['pruned_idxes']
                    ]
                    continue
                else:
                    curr_num_important_groups = len(group['important_idxes'])  # 当前重要组数量
                    trial_num_important_groups = curr_num_important_groups - len(group['active_redundant_idxes'])  # 试验后的重要组数量

                    if trial_num_important_groups % self.group_divisible != 0 or trial_num_important_groups <= 0:
                        ratio = trial_num_important_groups // self.group_divisible + 1
                        refined_num_important_groups = None # 初始化调整后的重要组数量

                        if ratio <= 1 or trial_num_important_groups == 0:
                            refined_num_important_groups = max(int(self.group_divisible), 1)
                        else:
                            # 否则，将调整后的重要组数量设置为 ratio * self.group_divisible。
                            refined_num_important_groups = max(int(ratio * self.group_divisible), int(self.group_divisible))

                        # 调整后的重要组数量不能超过该参数组的总单元数。
                        refined_num_important_groups = min(group['num_groups'], refined_num_important_groups)
                        
                        # refined_num_active_redundant_groups = 总组数 - 已永久剪枝的组数 - 调整后应保留的重要组数
                        refined_num_active_redundant_groups = group['num_groups'] - len(group['pruned_idxes']) - refined_num_important_groups

                        self.target_num_redundant_groups += (refined_num_active_redundant_groups - len(group['active_redundant_idxes']))

                        # 根据调整后的数量，截断（或扩展，但这里是截断）当前参数组的活动冗余索引列表。
                        group['active_redundant_idxes'] = group['active_redundant_idxes'][:refined_num_active_redundant_groups]
                group['important_idxes'] = [i for i in group['important_idxes'] if (i not in group['active_redundant_idxes'] and i not in group['pruned_idxes'])]

                
    def commit_redundant_idxes(self):
        """
        提交上一个或者当前剪枝周期识别出的冗余组索引。
        这个方法在每个剪枝周期的末尾被调用（或者在下一个剪枝周期开始时，处理上一个周期的结果）。
        它的主要作用是：
        1. 将本周期内被标记为“活动冗余”（active_redundant_idxes）的组的索引，
            正式添加到“已剪枝”（pruned_idxes）的组的索引列表中。这意味着这些组现在被认为是永久剪枝的。
        2. 清空当前参数组的“活动冗余”索引列表，为下一个剪枝周期做准备。
        3. 重新计算并更新当前参数组的“重要”（important_idxes）组的索引列表，排除所有已永久剪枝的组。
        4. 清空当前参数组的重要性分数字典，因为在下一个剪枝周期开始时会重新计算。
        """
        # 遍历优化器中的所有参数组。
        for group in self.param_groups:
            # 只处理那些可剪枝 (is_prunable=True) 且非辅助 (is_auxiliary=False) 的参数组。
            if group['is_prunable'] and not group['is_auxiliary']:
                group['pruned_idxes'].extend(group['active_redundant_idxes'].copy())
                group['active_redundant_idxes'].clear()
                group['important_idxes'] = [i for i in range(group['num_groups']) if i not in group['pruned_idxes']]
                group['importance_scores'].clear()

    def step(self, loss=None, closure=None):
        if closure is not None:
            # 如果提供了闭包函数，调用它以重新计算损失并获取其值。
            loss = closure()    # 计算损失
            
        self.num_steps += 1  # 优化器执行的总步数加1。
        
        # 第一阶段：计算梯度变体 (effective gradients)
        self.compute_grad_variant()
    
        # 第二阶段：剪枝决策
        # 检查是否达到了开始剪枝的步骤 (self.start_pruning_step)，是否尚未达到目标稀疏度 (not self.reach_target_group_sparsity())，
        # 以及当前的剪枝周期是否未超过总的剪枝周期数 (self.curr_pruning_period < self.pruning_periods)。
        if self.num_steps >= self.start_pruning_step and self.curr_pruning_period < self.pruning_periods:
            # 检查是否到达了一个新的剪枝周期的开始点。
            if (self.num_steps - self.start_pruning_step - 1) % self.pruning_period_duration == 0:
                # 1. 提交上一个剪枝周期识别的冗余组索引。
                self.commit_redundant_idxes()
                # 2. 为所有可剪枝的参数组计算新的重要性分数。
                self.compute_importance_scores()
                # 3. 根据新的重要性分数，识别当前剪枝周期要剪掉的冗余组。
                self.identify_redundant_groups()
                # 4. 进入下一个剪枝周期。
                self.curr_pruning_period += 1
        
        # 第三阶段：参数更新
        # t 表示当前剪枝周期内已经进行的步数。
        t = (self.num_steps - self.start_pruning_step) % self.pruning_period_duration
        for group in self.param_groups:
            if not group['is_prunable'] or len(group['active_redundant_idxes']) == 0:
                self.gradient_descent_step(group)
            elif group['is_prunable'] and len(group['active_redundant_idxes']) > 0:
                for (p_name, p, p_transform) in zip(group['p_names'], group['params'], group['p_transform']):
                    # 检查当前参数 (p_name) 是否在预先计算的梯度变体字典 (group['grad_variant']) 中。
                    # 如果一个参数没有梯度（例如，被冻结或在反向传播中未使用），它就不会有梯度变体。
                    if p_name not in group['grad_variant']:
                        # 如果没有梯度变体，则跳过该参数的更新。
                        continue
                    # 检查是否需要应用权重衰减，并且优化器变体是 'adamw'。
                    # AdamW 优化器的权重衰减是直接从参数中减去的，而不是像标准 L2 正则化那样加到梯度上。
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    # 使用预先计算好的梯度变体 (group['grad_variant'][p_name]) 和学习率 (group['lr']) 来更新参数。
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])

                    active_redundant_idxes = index_transformation_param_group(group['active_redundant_idxes'], p_transform, group)
                    if pruning_axis(p_transform) == 1 and len(p.data.shape) > 1:
                        p.data[:, active_redundant_idxes, ...] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)
                    else:
                        p.data[active_redundant_idxes] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)

                    # 处理辅助节点参数
                    for ng_id, offset in group['auxiliary_ngs']:
                        active_redundant_aux_idxes = [i + offset for i in active_redundant_idxes]
                        for aux_p in self.auxiliary_param_groups[ng_id]['params']:
                            if aux_p.grad is None:
                                continue
                            
                            # print(f"辅助参数组大小: aux_p.data.size(0)={aux_p.data.size(0)}, active_redundant_aux_idxes={active_redundant_aux_idxes}")
                            valid_idxes = [i for i in active_redundant_aux_idxes if 0 <= i < aux_p.data.size(0)]
                            if not valid_idxes:
                                # print(f"警告: 无有效索引，跳过更新。active_redundant_aux_idxes={active_redundant_aux_idxes}, aux_p.data.size(0)={aux_p.data.size(0)}")
                                continue
                            
                            aux_p.data[valid_idxes, ...] *= (self.pruning_period_duration - t - 1.0) / (self.pruning_period_duration - t)
            self.fix_pruned_groups_as_zeros(group)

        if self.num_steps >= self.start_pruning_step and t == self.pruning_period_duration - 1:
            # 提交当前周期识别的活动冗余组，将它们正式标记为已剪枝。
            self.commit_redundant_idxes()

