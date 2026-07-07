import torch
import numpy as np
from torch.optim.optimizer import required
from .base_hybrid_sparse_optimizer import BaseHybridSparseOptimizer
from only_train_once.transform import tensor_transformation, TensorTransform, index_transformation_param_group

class HESSOCRIC(BaseHybridSparseOptimizer):
    """
    HESSO-CRIC 优化器。
    CRIC (Conflicting Rules in Implementation Checking) 是一种通过迭代测试来识别真正可剪枝组的方法。
    它通过在多个周期内反复试探性地剪枝某些组，并观察其对模型性能（如损失）的影响，
    来确定哪些组是“违反约束”的（即重要的），哪些是冗余的。
    """
    def __init__(self, params, variant='sgd', lr=required, first_momentum=None, second_momentum=None, dampening=None, weight_decay=None, \
                 target_group_sparsity=0.5, tolerance=0, group_divisible=1, \
                 start_cric_step=0, max_cycle_period=10, sampling_steps=None, hybrid_training_steps=None, \
                 importance_score_criteria='default'):
        """
        HESSOCRIC 优化器的构造函数。

        参数:
            ... (标准优化器参数) ...
            target_group_sparsity (float): 最终的目标组稀疏度。
            tolerance (int): CRIC 终止条件之一，当违反约束的组数量小于等于此值时，CRIC 过程可以提前结束。
            group_divisible (int): 确保剪枝后剩余的组数量是此值的整数倍。
            start_cric_step (int): 训练的第几步开始 CRIC 过程。
            max_cycle_period (int): CRIC 过程最多执行多少个周期。
            sampling_steps (int): 每个 CRIC 周期内用于采样和评估的训练步数。
            hybrid_training_steps (int): CRIC 结束后，混合训练（逐渐冻结冗余组）的持续步数。
            importance_score_criteria (dict or str): 定义各种重要性分数加权求和时的权重。
        """

        print("Setup HESSOCRIC")
        # CRIC 过程的起始步骤
        self.start_cric_step = start_cric_step
        # 每个 CRIC 周期包含的采样步数
        self.sampling_steps = sampling_steps 
        # CRIC 过程的最大周期数
        self.max_cycle_period = int(max(1, max_cycle_period)) 
        # 当前所在的 CRIC 周期，-1 表示尚未开始
        self.curr_cycle_period = -1 
        # CRIC 结束后混合训练的步数
        self.hybrid_training_steps = hybrid_training_steps

        # 标记是否已经识别出所有冗余组
        self.redundant_groups_identified = False
        # CRIC 终止的容忍度，即允许的最小违反组数量
        self.tolerance = tolerance

        # 设置重要性分数的加权标准
        if importance_score_criteria == 'default' or importance_score_criteria is None:
            # 如果使用默认设置，为各种重要性指标分配权重
            self.importance_score_criteria = {'magnitude': 0.2, 'avg_magnitude': 0.2,\
                                              'cosine_similarity': 0.2, \
                                              'taylor_first_order': 0.2, 'taylor_second_order': 0.2, 'loss': 1.0}
        else:
            # 否则使用用户自定义的权重
            self.importance_score_criteria = importance_score_criteria

        # 调用父类的构造函数，完成基础优化器的初始化
        super(HESSOCRIC, self).__init__(params=params, variant=variant, lr=lr, first_momentum=first_momentum, second_momentum=second_momentum, \
                                        dampening=dampening, weight_decay=weight_decay, target_group_sparsity=target_group_sparsity, \
                                        group_divisible=group_divisible)

        # 为每个参数组初始化用于 CRIC 的各种索引列表和状态字典
        for param_group in self.param_groups:
            # 'important_idxes': 重要组的索引，初始时所有组都是重要的
            param_group['important_idxes'] = [i for i in range(param_group['num_groups'])]
            # 'active_violating_idxes': 当前 CRIC 周期中被识别为“违反约束”（即重要）的组
            param_group['active_violating_idxes'] = list()
            # 'trial_violating_idxes': 在试探阶段被临时标记为可能违反约束的组
            param_group['trial_violating_idxes'] = list()
            # 'historical_violating_idxes': 在所有历史周期中被确认是违反约束的组
            param_group['historical_violating_idxes'] = list()
            # 'active_redundant_idxes': CRIC 结束后，被识别为冗余但在混合训练阶段尚未完全冻结的组
            param_group['active_redundant_idxes'] = list()
            # 'pruned_idxes': 最终被确认并剪枝（权重清零并冻结）的组
            param_group['pruned_idxes'] = list()
            # 'importance_scores': 存储当前计算出的各种重要性分数
            param_group['importance_scores'] = dict()
            
        print("Total number of groups")
        print(self.target_group_sparsity, self.total_num_groups, self.target_num_redundant_groups)

        # 创建参数缓存字典，用于在 CRIC 试探过程中保存和恢复模型权重
        self.cache_parameters = dict()
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                # 克隆当前参数权重作为备份
                self.cache_parameters[p_name] = param.data.clone()        
            # 只为可剪枝的参数组初始化用于收集信息的字典
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                # 'importance_score_collection': 收集每个 CRIC 周期的重要性分数样本
                param_group['importance_score_collection'] = dict()
                # 'active_violating_idxes_collection': 收集每个 CRIC 周期的违反组信息
                param_group['active_violating_idxes_collection'] = dict()
                # 'loss_collection': 收集每个 CRIC 周期的损失变化信息
                param_group['loss_collection'] = dict()
                # 为每个可能的周期初始化空列表
                for cycle_period in range(self.max_cycle_period + 1):
                    param_group['importance_score_collection'][cycle_period] = list()
                    param_group['active_violating_idxes_collection'][cycle_period] = list()
                    param_group['loss_collection'][cycle_period] = list()

        # 创建一个只包含可剪枝参数组的字典，方便快速访问
        self.prunable_param_group_dict = dict()
        for param_group in self.param_groups:
            if param_group['is_prunable'] and not param_group['is_auxiliary']:
                self.prunable_param_group_dict[param_group['id']] = param_group
        # 获取所有可剪枝参数组的 ID
        self.param_group_ids = list(self.prunable_param_group_dict.keys())
        # 定义在 CRIC 初始阶段要测试的几种稀疏度
        self.trial_group_sparsties = [0.25, 0.5, 0.75]
        # 计算全局采样阶段的起始步骤
        self.start_global_sampling_step = 2 * len(self.trial_group_sparsties) * len(self.prunable_param_group_dict) + self.start_cric_step

        # 标记 CRIC 过程是否已经终止
        self.is_cric_terminated = False

    def reset_cache_params(self):
        """将当前模型参数更新到缓存中。"""
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                self.cache_parameters[p_name] = param.data.clone()  

    def reset_params(self):
        """从缓存中恢复模型参数，并将梯度清零。"""
        for param_group in self.param_groups:
            for (p_name, param) in zip(param_group['p_names'], param_group['params']):
                # 将参数数据恢复到缓存的状态
                param.data.copy_(self.cache_parameters[p_name]) 
                # 如果参数有梯度，则将其清零
                if param.requires_grad and param.grad is not None:
                    param.grad.zero_()

    def compute_num_active_violating_groups(self):
        """计算当前所有参数组中“活跃违反组”的总数。"""
        num_violating_groups = 0
        for param_group in self.param_groups:
            num_violating_groups += len(param_group['active_violating_idxes'])
        return num_violating_groups

    def cric_terminate(self):
        """
        判断 CRIC 过程是否应该终止。
        终止条件：
        1. 当前周期数已达到设定的最大周期数。
        2. 当前周期数大于等于1，并且活跃违反组的数量小于等于容忍度。
        """
        if self.curr_cycle_period >= self.max_cycle_period:
            return True
        elif self.curr_cycle_period >= 1 and self.compute_num_active_violating_groups() <= self.tolerance:
            return True
        else:
            return False

    def update_violating_set(self, cycle_period=0):
        """
        更新当前周期的“违反约束组”集合 (active_violating_idxes)。
        这个集合代表了在本轮 CRIC 迭代中被识别为重要的组。
        """
        print("Update violating set")
        # 将所有参数组的重要性分数拼接成一个全局分数张量
        global_scores = torch.cat(self.global_scores, dim=0)
        # 确定要识别为“违反约束”的组的数量，这通常等于目标冗余组的数量
        curr_K = self.target_num_redundant_groups

        # 找到全局分数中最小的 K 个值的索引（因为分数越小越不重要，所以取负号后找最大的K个）
        # 这些索引对应全局最不重要的组，即潜在的“违反约束组”
        _, top_indices = torch.topk(-global_scores, curr_K)
        top_indices = top_indices.cpu().numpy().tolist()
    
        # 遍历每个参数组
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                # 如果是第一个 CRIC 周期 (cycle_period == 1)
                if cycle_period == 1:
                    # 直接使用全局 top_indices 来确定该组的违反约束组
                    # 计算全局索引与当前组全局索引范围的交集
                    global_trial_violating_idxes = np.intersect1d(top_indices, group['global_idxes'])
                    # 将全局索引转换为组内局部索引
                    group['active_violating_idxes'] = (global_trial_violating_idxes - group['global_start_idx']).tolist()
                else:
                    # 对于后续周期，从“试探性违反组”中筛选出尚未被历史确认的组，作为当前周期的活跃违反组
                    group['active_violating_idxes'] = [i for i in group['trial_violating_idxes'] if i not in group['historical_violating_idxes']]
                # 更新重要组列表：所有组中排除了当前活跃的违反约束组
                group['important_idxes'] = [i for i in range(group['num_groups']) if i not in group['active_violating_idxes']]

    def update_trial_violating_set(self):
        """
        更新“试探性违反约束组”集合 (trial_violating_idxes)。
        这个集合包含了根据当前重要性分数可能被认为是重要的组。
        """
        # 将所有参数组的重要性分数拼接成一个全局分数张量
        global_scores = torch.cat(self.global_scores, dim=0)
        # 确定要识别的组的数量
        curr_K = self.target_num_redundant_groups

        # 找到全局最不重要的 K 个组的索引
        _, top_indices = torch.topk(-global_scores, curr_K)
        top_indices = top_indices.cpu().numpy().tolist()

        # 遍历每个参数组
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                # 计算全局最不重要索引与当前组索引的交集
                global_trial_violating_idxes = np.intersect1d(top_indices, group['global_idxes'])
                # 将新识别出的局部索引添加到试探性违反组列表中
                group['trial_violating_idxes'] += (global_trial_violating_idxes - group['global_start_idx']).tolist()
                # 筛选掉那些已经被当前周期或历史周期确认的违反组
                group['trial_violating_idxes'] = [i for i in group['trial_violating_idxes'] if i not in group['active_violating_idxes'] \
                                                  and i not in group['historical_violating_idxes']]
                # 去重，确保列表中的索引是唯一的
                group['trial_violating_idxes'] = list(set(group['trial_violating_idxes']))

    def update_historical_violating_set(self):
        """
        更新“历史违反约束组”集合 (historical_violating_idxes)。
        将当前周期确认的违反组永久性地记录下来。
        """
        for param_group in self.param_groups:
            # 将当前活跃的违反组添加到历史列表中
            param_group['historical_violating_idxes'] += param_group['active_violating_idxes']
            # 去重，保持历史列表的唯一性
            param_group['historical_violating_idxes'] = list(set(param_group['historical_violating_idxes']))

    def proj_trial_group_sparsity(self, param_group, trial_group_sparsity):
        """
        试探性地将一个参数组投影（剪枝）到给定的稀疏度。
        这会直接修改参数的权重，将最不重要的组清零。
        """
        # 如果该组没有重要性分数，则无法进行投影
        if 'importance_scores' not in param_group:
            return

        # 根据试探稀疏度计算需要剪枝的组的数量
        num_redund_grps = max(min(int(param_group['num_groups'] * trial_group_sparsity), param_group['num_groups']), 1)
        # 根据总体重要性分数，找到最不重要的组的索引
        _, proj_indices = torch.topk(-param_group['importance_scores']['overall'], num_redund_grps)
        proj_indices = proj_indices.cpu().numpy().tolist()
        # 遍历该组中的所有参数张量及其变换方式
        for (p, p_transform) in zip(param_group['params'], param_group['p_transform']):
            # 对索引进行可能的变换（例如，对于多头注意力的 head dim）
            proj_indices = index_transformation_param_group(proj_indices, p_transform, param_group)
            # 根据张量的形状和变换方式，将对应索引的权重清零
            if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                # 对于需要转置处理的二维以上张量（如某些线性层）
                p.data[:, proj_indices, ...] = 0.0
            else:
                # 对于标准情况（如卷积核的输出通道）
                p.data[proj_indices] = 0.0

    def cric_step(self):
        """
        执行一个 CRIC (Conflicting Rules in Implementation Checking) 步骤。
        这个函数在全局采样阶段被调用，用于识别重要的和冗余的参数组。
        """
        # print("Current sampling period", self.curr_sampling_period, self.num_steps)
        # 首先，计算当前步骤的各种重要性分数
        self.compute_importance_scores()
        # 将计算出的重要性分数提交并保存到当前 CRIC 周期的数据收集中
        self.commit_important_scores(self.curr_cycle_period)

        # 检查是否到达了一个新的 CRIC 周期的开始
        if (self.num_steps - self.start_global_sampling_step) % self.sampling_steps == 0:
            # 如果是，则当前 CRIC 周期数加一
            self.curr_cycle_period += 1
            # 根据累积的重要性分数，更新当前周期的“违反约束组”
            self.update_violating_set(self.curr_cycle_period)
            # 将当前周期识别出的违反组添加到历史记录中
            self.update_historical_violating_set()
            # 重置模型参数到本周期开始前的状态，以便进行下一次干净的评估
            self.reset_params()
        # 更新“试探性违反组”列表，为下一个周期的决策做准备
        self.update_trial_violating_set()
        
        # 第二遍：更新参数权重
        # t 表示在当前采样周期内已经过的步数
        t = (self.num_steps - self.start_global_sampling_step - 1) % self.sampling_steps
        for group in self.param_groups:
            # 跳过不可剪枝的或辅助的参数组
            if not (group['is_prunable'] and not group['is_auxiliary']):
                continue
            # 如果当前组没有活跃的违反约束组（即所有组都被认为是重要的）
            if len(group['active_violating_idxes']) == 0:
                # 对该组的所有参数执行标准的梯度下降步骤
                for p_name, p in zip(group['p_names'], group['params']):
                    if p_name not in group['grad_variant']:
                        continue
                    # 如果使用 AdamW，先应用权重衰减
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    # 应用梯度更新
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
            else:
                # 如果存在活跃的违反约束组
                for (p_name, p, p_transform) in zip(group['p_names'], group['params'], group['p_transform']):
                    if p_name not in group['grad_variant']:
                        continue
                    # 先执行标准的梯度下降步骤
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
                    
                    # 对被识别为“违反约束”的组的权重进行缩放
                    # 这个缩放操作是为了逐渐减小这些组的权重，模拟剪枝效果
                    scaling_factor = (self.sampling_steps - t - 1.0) / (self.sampling_steps - t)
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, group['active_violating_idxes'], ...] *= scaling_factor
                    else:
                        p.data[group['active_violating_idxes']] *= scaling_factor

                    # 处理与该组相关的辅助参数（例如 BatchNorm）
                    for ng_id, offset in group['auxiliary_ngs']:
                        aux_pg = self.auxiliary_param_groups[ng_id]
                        for aux_p in aux_pg['params']:
                            if aux_p.grad is None:
                                continue
                            # 对辅助参数中对应的部分也进行同样的缩放
                            aux_p.data[offset:offset+group['num_groups'], ...] *= scaling_factor

    def basic_step(self):
        """
        执行一个基本的优化器步骤（标准的梯度下降）。
        这在 CRIC 过程开始前使用。
        """
        for param_group in self.param_groups:
            self.gradient_descent_step(param_group)

    def proj_step(self, loss):
        """
        执行一个投影步骤。
        在 CRIC 的初始阶段，交替地将参数组投影（剪枝）到不同的稀疏度，并评估损失变化。
        """
        # 计算并保存当前的重要性分数
        self.compute_importance_scores()
        self.commit_important_scores(self.curr_cycle_period)

        # 根据当前总步数，确定正在处理哪个参数组和哪个试探稀疏度
        curr_param_group_idx = (self.num_steps - self.start_cric_step) // (2 * len(self.trial_group_sparsties))
        curr_trial_group_sparsity_idx = (self.num_steps - self.start_cric_step) // 2 % len(self.trial_group_sparsties)
        # 判断当前是执行投影还是收集损失（交替进行）
        do_proj = (self.num_steps - self.start_cric_step) % 2 == 0
        curr_param_group = self.prunable_param_group_dict[self.param_group_ids[curr_param_group_idx]]
        curr_trial_group_sparsity = self.trial_group_sparsties[curr_trial_group_sparsity_idx]

        if do_proj:
            # 如果是投影步骤，则将当前参数组剪枝到目标稀疏度
            self.proj_trial_group_sparsity(curr_param_group, curr_trial_group_sparsity)
        else:
            # 如果是收集损失步骤，记录投影后的损失相对于参考损失的变化
            curr_param_group['loss_collection'][self.curr_cycle_period].append(loss.item() / self.ref_loss) 
            # 恢复参数到投影前的状态
            self.reset_params()
        # 记录下在这次投影测试中，哪些组被认为是违反约束的（在这里是所有组，作为基线）
        curr_param_group['active_violating_idxes_collection'][self.curr_cycle_period] = [i for i in range(curr_param_group['num_groups'])]
        

    def hybrid_step(self):
        """
        执行混合训练步骤。
        在 CRIC 过程终止后，此步骤用于逐渐将已识别的冗余组的权重衰减至零，
        而不是立即将它们剪掉，从而实现更平滑的过渡。
        """
        # t: 当前在混合训练阶段中已经过的步数
        t = self.num_steps - self.cric_terminated_step - 1
        for group in self.param_groups:
            # 条件：如果组不可剪枝，或没有活动的冗余组，或混合训练阶段已结束
            # 在这些情况下，执行标准的梯度下降步骤
            if not group['is_prunable'] or len(group['active_redundant_idxes']) == 0 or self.num_steps > self.cric_terminated_step + self.hybrid_training_steps:
                # 对该组的所有参数执行标准的梯度下降步骤
                for p_name, p in zip(group['p_names'], group['params']):
                    if p_name not in group['grad_variant']:
                        continue
                    # 如果使用 AdamW，先应用权重衰减
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    # 应用梯度更新
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])
            # 条件：如果组是可剪枝的，并且存在活动的冗余组（即处于混合训练阶段）
            elif group['is_prunable'] and len(group['active_redundant_idxes']) > 0:
                # 遍历组内所有参数
                for (p_name, p, p_transform) in zip(group['p_names'], group['params'], group['p_transform']):
                    if p_name not in group['grad_variant']:
                        continue
                    # 首先，对所有参数执行标准的梯度下降步骤
                    if group['weight_decay'] is not None and group['variant'] == 'adamw':
                        p.data.add_(group['weight_decay'] * p.data, alpha=-group['lr'])
                    p.data.add_(group['grad_variant'][p_name], alpha=-group['lr'])

                    # 对冗余组的索引进行可能的变换（例如，适应多头注意力的 head dim）
                    active_redundant_idxes = index_transformation_param_group(group['active_redundant_idxes'], p_transform, group)
                    # 计算一个缩放因子，该因子会随着 t 的增加从接近 1 逐渐变为 0
                    scaling_factor = (self.hybrid_training_steps - t - 1.0) / (self.hybrid_training_steps - t)
                    # 对冗余组的权重应用缩放，使其逐渐衰减
                    if p_transform == TensorTransform.TRANSPOSE and len(p.data.shape) > 1:
                        p.data[:, active_redundant_idxes, ...] *= scaling_factor
                    else:
                        p.data[active_redundant_idxes] *= scaling_factor
                    
                    # 处理相关的辅助参数（例如 BatchNorm）
                    for ng_id, offset in group['auxiliary_ngs']:
                        # 计算辅助参数中对应的冗余索引
                        active_redundant_aux_idxes = [i + offset for i in active_redundant_idxes]
                        for aux_p in self.auxiliary_param_groups[ng_id]['params']:
                            if aux_p.grad is None:
                                continue
                            # 对辅助参数的相应部分也应用同样的衰减
                            aux_p.data[active_redundant_aux_idxes, ...] *= scaling_factor

            # 当混合训练的最后一步完成时
            if self.num_steps == self.cric_terminated_step + self.hybrid_training_steps:
                # 将活动的冗余组正式移动到“已剪枝”列表中
                group['pruned_idxes'].extend(group['active_redundant_idxes'])
                # 清空活动的冗余组列表
                group['active_redundant_idxes'].clear()

            # 在混合训练阶段结束后
            if self.num_steps > self.cric_terminated_step + self.hybrid_training_steps:
                # 确保所有被标记为“已剪枝”的组的权重和梯度都固定为零
                self.fix_pruned_groups_as_zeros(group)

    def step(self, loss=None, closure=None):
        """
        执行一个优化步骤。这是优化器的主要入口点。
        它根据当前的训练阶段（预热、CRIC、混合训练）调用不同的内部方法。
        """
        # 标准的 PyTorch 优化器模式，允许通过闭包重新计算损失
        if closure is not None:
            loss = closure()

        # 增加总步数计数器
        self.num_steps += 1

        # 第一步：计算梯度变体（例如，对于 Adam 是动量和二阶矩）
        self.compute_grad_variant()

        # 当训练到达 CRIC 开始的步骤时，进行初始化
        if self.num_steps == self.start_cric_step:
            # 记录参考损失，用于后续的损失变化评估
            self.ref_loss = loss.item() if loss is not None else None
            # 缓存当前模型参数，作为恢复点
            self.reset_cache_params()
            # 开始第一个 CRIC 周期
            self.curr_cycle_period += 1 

        # 根据当前步数，进入不同的训练阶段
        if self.num_steps < self.start_cric_step:
            # CRIC 开始前：执行标准的梯度下降
            self.basic_step()
        elif self.num_steps >= self.start_cric_step and self.num_steps < self.start_global_sampling_step:
            # CRIC 初始阶段：执行投影测试步骤
            self.proj_step(loss)
        elif self.num_steps >= self.start_global_sampling_step and self.curr_cycle_period < self.max_cycle_period and not self.is_cric_terminated: 
            # CRIC 全局采样阶段：执行核心的 CRIC 步骤
            self.cric_step()
        elif self.is_cric_terminated:
            # CRIC 结束后：执行混合训练步骤
            self.hybrid_step()
        
        # 检查 CRIC 过程是否应该终止
        if not self.is_cric_terminated and self.cric_terminate():
            print("cric_terminate", self.num_steps)
            # 计算累积的重要性分数
            self.compute_accumulate_saliency_score()
            # 根据累积的分数，最终确定哪些组是冗余的
            self.identify_redundant_groups()
            # 将参数恢复到 CRIC 开始前的状态
            self.reset_params()
            # 设置终止标志
            self.is_cric_terminated = True
            # 记录 CRIC 终止的步骤
            self.cric_terminated_step = self.num_steps
            # TODO: 可以在这里移除一些不再需要的信息以节省内存
        return

    def compute_accumulate_saliency_score(self):
            """
            计算累积的重要性分数。
            此方法在 CRIC 过程结束时调用，它整合了在所有 CRIC 周期中收集到的信息，
            以得出一个最终的、更鲁棒的重要性排名。
            """
            # 遍历所有参数组
            for param_group in self.param_groups:
                if param_group['is_prunable'] and not param_group['is_auxiliary']:
                    # 初始化累积重要性分数和计数器
                    param_group['accumulated_importance_score'] = None
                    accumulate_count = 0
                    # 遍历每个 CRIC 周期收集到的重要性分数样本
                    for cycle_period in param_group['importance_score_collection']:
                        importance_score_sample_steps = param_group['importance_score_collection'][cycle_period]
                        # 累加每个样本的'overall'（总体）重要性分数
                        for importance_score in importance_score_sample_steps:
                            if param_group['accumulated_importance_score'] is None:
                                param_group['accumulated_importance_score'] = importance_score['overall'].clone()
                            else:
                                param_group['accumulated_importance_score'] += importance_score['overall']
                            accumulate_count += 1
                    # 如果收集到了分数，则计算平均重要性分数
                    if param_group['accumulated_importance_score'] is not None:
                        param_group['accumulated_importance_score'] /= float(accumulate_count)

                    # 遍历每个 CRIC 周期收集到的“违反约束组”和对应的损失变化信息
                    for cycle_period in param_group['active_violating_idxes_collection']:
                        if len(param_group['active_violating_idxes_collection'][cycle_period]) > 0:
                            violating_idxes = param_group['active_violating_idxes_collection'][cycle_period]
                            loss_scores = param_group['loss_collection'][cycle_period]
                            if len(loss_scores) == 0:
                                continue
                            # 计算平均损失分数，并通过违反组的数量进行归一化
                            # 这是为了将“剪掉后损失增加”这一信息也融入最终的重要性分数中
                            avg_loss_score = sum(loss_scores) / len(loss_scores) / len(violating_idxes)
                            # 将加权后的损失分数加到对应组的累积重要性分数上
                            param_group['accumulated_importance_score'][violating_idxes] += self.importance_score_criteria['loss'] * avg_loss_score

    def identify_redundant_groups(self):
        """
        根据累积的重要性分数，最终确定哪些组是冗余的（即可以被剪枝）。
        """
        accumulated_global_scores = list()
        # 将所有可剪枝参数组的累积重要性分数收集起来
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                accumulated_global_scores.append(group['accumulated_importance_score'])
        
        # 将所有分数拼接成一个全局张量
        accumulated_global_scores = torch.cat(accumulated_global_scores, dim=0)
        # 在全局范围内，找到分数最低的 K 个组，K 等于目标冗余组的数量
        _, top_indices = torch.topk(-accumulated_global_scores, self.target_num_redundant_groups)
        top_indices = top_indices.cpu().numpy()

        # 遍历每个参数组，分配最终的冗余组
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                # 找到全局冗余索引与当前组索引的交集，确定本组的冗余组
                global_active_redundant_idx = np.intersect1d(top_indices, group['global_idxes'])
                group['active_redundant_idxes'] = (global_active_redundant_idx - group['global_start_idx']).tolist()
                
                # --- 关键步骤：根据 group_divisible 约束来优化剪枝数量 ---
                # 这是为了确保剪枝后剩余的通道数（组数）能被特定数值整除，以利于硬件加速
                if group['num_groups'] < self.group_divisible:
                    # 如果总组数小于约束值，则不进行剪枝
                    group['active_redundant_idxes'].clear()
                    group['pruned_idxes'].clear()
                else:
                    # 计算试探性剪枝后剩余的重要组数量
                    curr_num_important_groups = len(group['important_idxes'])
                    trial_num_important_groups = curr_num_important_groups - len(group['active_redundant_idxes'])                    
                    # 如果剩余组数不满足整除约束，或者小于等于0
                    if trial_num_important_groups % self.group_divisible != 0 or trial_num_important_groups <= 0:
                        # 向上取整，计算需要保留的组数应该是 group_divisible 的哪个倍数
                        ratio = trial_num_important_groups // self.group_divisible + 1 # +1 表示倾向于保留更多组
                        refined_num_important_groups = None
                        if ratio <= 1 or trial_num_important_groups == 0:
                            # 至少保留 group_divisible 个组
                            refined_num_important_groups = max(int(self.group_divisible), 1)
                        else:
                            # 保留 ratio * group_divisible 个组
                            refined_num_important_groups = max(int(ratio * self.group_divisible), int(self.group_divisible))
                        # 确保保留的组数不超过总组数
                        refined_num_important_groups = min(group['num_groups'], refined_num_important_groups)
                        # 根据优化后的保留组数，重新计算需要剪掉的冗余组数量
                        refined_num_active_redundant_groups = group['num_groups'] - len(group['pruned_idxes']) - refined_num_important_groups
                        # 更新全局的目标冗余组数量（因为局部调整了）
                        self.target_num_redundant_groups += (refined_num_active_redundant_groups - len(group['active_redundant_idxes']))
                        # 截断冗余组列表，以匹配优化后的数量
                        group['active_redundant_idxes'] = group['active_redundant_idxes'][:refined_num_active_redundant_groups]
                # 最终更新重要组列表，从中移除被确定为冗余和已剪枝的组
                group['important_idxes'] = [i for i in group['important_idxes'] if (i not in group['active_redundant_idxes'] and i not in group['pruned_idxes'])]

    def commit_important_scores(self, cycle_period):
        """
        将当前计算出的重要性分数保存到对应周期的收集中。
        """
        for group in self.param_groups:
            if group['is_prunable'] and not group['is_auxiliary']:
                # 将当前的重要性分数字典追加到指定周期的列表中
                group['importance_score_collection'][cycle_period].append(group['importance_scores'])


    def compute_metrics(self):
        """
        计算并更新优化器在当前状态下的各种指标。
        这些指标用于监控剪枝过程，例如组稀疏度、不同类型组的数量和范数等。
        """
        # --- 初始化/重置所有指标 ---
        # 总参数范数（所有组的L2范数之和）
        self.opt_metrics.norm_params = 0.0
        # 重要组的L2范数之和
        self.opt_metrics.norm_important_groups = 0.0
        # 冗余组的L2范数之和
        self.opt_metrics.norm_redundant_groups = 0.0
        # 违反约束组的L2范数之和
        self.opt_metrics.norm_violating_groups = 0.0
        # 权重为零的组的数量
        self.opt_metrics.num_zero_groups = 0
        # 重要组的数量
        self.opt_metrics.num_important_groups = 0
        # 冗余组的数量（包括正在混合训练中和已剪枝的）
        self.opt_metrics.num_redundant_groups = 0
        # 当前活跃的违反约束组的数量
        self.opt_metrics.num_violating_groups = 0
        # 试探性违反约束组的数量
        self.opt_metrics.num_trial_violating_groups = 0
        # 历史违反约束组的总数
        self.opt_metrics.num_historical_violating_groups = 0
        
        # --- 遍历参数组计算指标 ---
        for group in self.param_groups:
            # 只处理可剪枝的、非辅助的参数组
            if not (group['is_prunable'] and not group['is_auxiliary']):
                continue
            
            # 用于累加组内所有参数张量的范数
            norm_group = None
            # 获取当前组中不同类别的索引列表
            import_idxes = group['important_idxes']
            redund_idxes = group['active_redundant_idxes'] + group['pruned_idxes']
            violat_idxes = group['active_violating_idxes']

            # 遍历组内的每个参数张量（一个组可能包含多个张量，如Conv层和其bias）
            for param, p_transform in zip(group['params'], group['p_transform']):
                # 跳过明确标记为不剪枝的参数
                if p_transform == TensorTransform.NO_PRUNE:
                    continue
                
                # 对参数张量进行变换，使其形状适合按组计算范数
                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    # 特殊处理多头注意力机制的 head dimension
                    param_transform = tensor_transformation(param.data, p_transform, group['num_groups'], group['num_heads'])
                else:
                    # 标准变换
                    param_transform = tensor_transformation(param.data, p_transform, group['num_groups'])
                
                # 计算每个组的L2范数的平方，并累加
                if norm_group is None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2
            
            # 开方得到最终的L2范数
            norm_group = torch.sqrt(norm_group)

            # --- 累加各项指标 ---
            # 统计范数为0的组的数量
            self.opt_metrics.num_zero_groups += torch.sum(norm_group == 0).item()
            # 累加所有组的范数
            self.opt_metrics.norm_params += torch.sum(norm_group).item()
            # 累加重要组的范数
            self.opt_metrics.norm_important_groups += torch.sum(norm_group[import_idxes]).item()
            # 累加冗余组的范数
            self.opt_metrics.norm_redundant_groups += torch.sum(norm_group[redund_idxes]).item()
            # 累加违反约束组的范数
            self.opt_metrics.norm_violating_groups += torch.sum(norm_group[violat_idxes]).item()
            # 累加各类组的数量
            self.opt_metrics.num_important_groups += len(import_idxes)
            self.opt_metrics.num_redundant_groups += len(redund_idxes)
            self.opt_metrics.num_violating_groups += len(violat_idxes)
            self.opt_metrics.num_trial_violating_groups += len(group['trial_violating_idxes'])
            self.opt_metrics.num_historical_violating_groups += len(group['historical_violating_idxes'])

        # 计算最终的组稀疏度 = (零值组的数量 / 总组数)
        self.opt_metrics.group_sparsity = self.opt_metrics.num_zero_groups / float(self.total_num_groups + self.safe_guard)

        # 返回包含所有计算好的指标的对象
        return self.opt_metrics

