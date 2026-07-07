from .graph import Graph  # 图结构类，用于构建模型节点关系图
from .dependency_graph import build_pruning_dependency_graph  # 构建剪枝依赖图的函数
from .subnet_construction import automated_pruning_compression  # 自动剪枝压缩子网的函数
import os  # 操作系统接口模块，用于文件路径等操作


class OTO:
    """
    One-Time Optimization (OTO) 类：提供一次性剪枝压缩训练能力。
    核心作用是将神经网络模型构建成一个图（Graph），进行剪枝分析，并支持多种优化器进行稀疏化处理，
    最终生成可部署的子网络模型。
    """
    def __init__(self, model=None, dummy_input=None, compress_mode='prune', skip_patterns=None, strict_out_nodes=False):
        """
        初始化 OTO 实例。
        参数：
            model: 待压缩模型对象。
            dummy_input: 模型输入样例（用于构建计算图）。
            compress_mode: 压缩模式，默认为 'prune'（剪枝），也可为 'erase'（未来支持）。
            skip_patterns: 需跳过处理的节点/层模式。
            strict_out_nodes: 是否严格对待输出节点。
        """
        self._graph = None                              # 存储构建好的图结构
        self._model = model                             # 原始模型
        self._dummy_input = dummy_input                 # 输入样本
        self._skip_patterns = skip_patterns             # 跳过模式
        self._strict_out_nodes = strict_out_nodes       # 输出节点是否严格处理，是否严格只将无出边节点作为输出节点
        self._mode = compress_mode                      # 压缩模式
        # 如果提供了模型和输入，则初始化图结构
        if self._model is not None and self._dummy_input is not None:
            # 初始化模型的计算图（Graph），为后续的剪枝、优化和稀疏化操作提供基础。
            print("正在创建计算图...")
            self.initialize(model=self._model, dummy_input=self._dummy_input, skip_patterns=self._skip_patterns, strict_out_nodes=self._strict_out_nodes)
            print("计算图创建完成！")
            if self._mode == 'prune':
                # 执行剪枝相关流程
                self.partition_pzigs()                  # 构建剪枝依赖图
                self.set_trainable()                    # 设置可训练参数
                self._graph.cluster_node_groups()       # 聚合节点组（按照求分组聚类）
            elif self._mode == 'erase':
                # 擦除模式暂未实现
                raise NotImplementedError

        self.compressed_model_path = None           # 压缩后模型保存路径
        self.full_group_sparse_model_path = None    # 全组稀疏模型保存路径

    def cluster_node_groups(self, num_clusters=1):
        """
        将图中节点按某种策略分组聚类。
        参数：
            num_clusters: 目标聚类数。
        """
        self._graph.cluster_node_groups(num_clusters=num_clusters)

    def initialize(self, model=None, dummy_input=None, skip_patterns=None, strict_out_nodes=False):
        """
        初始化图结构。
        参数：
            model: 神经网络模型。
            dummy_input: 模型输入样例。
            skip_patterns: 要跳过的节点/层模式。
            strict_out_nodes: 是否严格处理输出节点。
        """
        # 将模型切换到评估模式（eval），确保在构建计算图时不会受到训练模式（如 Dropout 或 BatchNorm）的影响。
        model = model.eval()
        self._model = model 
        self._dummy_input = dummy_input
        # 构建模型的计算图
        self._graph = Graph(model, dummy_input, skip_patterns=skip_patterns, strict_out_nodes=strict_out_nodes)

    def partition_pzigs(self):
        """
        构建剪枝依赖图（Pruning Dependency Graph）。
        pzigs 表示 pruning zigzag structure（剪枝锯齿结构），用于描述剪枝时各层之间的依赖关系。
        """
        build_pruning_dependency_graph(self._graph)

    def visualize(self, out_dir=None, view=False, vertical=True, by_node_groups=True, display_params=False, display_flops=False):
        """
        可视化当前模型的图结构（以 dot 文件格式）。
        参数：
            out_dir: 输出目录。
            view: 是否打开可视化结果。
            vertical: 图方向是否垂直。
            by_node_groups: 是否按节点组划分显示。
            display_params: 是否显示参数信息。
        """
        '''
        # 调用 Graph 对象的 build_dot 方法来构建一个 DOT 语言描述的图。
        # vertical: 控制图的布局方向，True 为垂直布局，False 为水平布局。
        # by_node_groups: 控制是否按节点组（NodeGroup）来组织和显示图结构。
        # display_params: 控制是否在图中显示节点的参数信息。
        # .render() 方法将生成的 DOT 图渲染成文件并保存。
        # os.path.join 用于构建输出文件的完整路径。
        # out_dir: 指定输出目录。如果为 None，则默认为当前目录 './'。
        # 文件名部分：
        #   - 如果模型对象 self._model 有 'name' 属性，则使用 self._model.name。
        #   - 否则，使用模型的类名 type(self._model).__name__。
        #   - 最后追加 '_pruning_dependency' 作为文件名后缀，表明这是剪枝依赖图。
        # view: 控制是否在渲染完成后自动打开生成的文件（通常是图片或PDF）。
        '''
        self._graph.build_dot(vertical=vertical, by_node_groups=by_node_groups, display_params=display_params, display_flops=display_flops).render(\
            os.path.join(out_dir if out_dir is not None else './', \
                self._model.name if hasattr(self._model, 'name') else type(self._model).__name__ + '_pruning_dependency'), \
                view=view)

    def hesso(self, lr=0.1, weight_decay=None, first_momentum=None, second_momentum=None, \
               variant='sgd', target_group_sparsity=0.5, start_pruning_step=0, \
               pruning_steps=1, pruning_periods=1, device='cuda',\
               dampening=None, group_divisible=1, fixed_zero_groups=True, importance_score_criteria='default', alpha_scheduler_cfg=None, seed=None):
        """
        初始化并返回 HESSO (Hessian-based Structured Sparsity Optimization) 优化器实例。
        HESSO 是一种用于结构化剪枝的优化器，它利用二阶信息（Hessian）来评估参数组的重要性，
        并逐步将不重要的参数组稀疏化（剪枝）。
        参数:
            lr (float): 学习率。默认为 0.1。
            weight_decay (float, optional): 权重衰减（L2 正则化）。默认为 None。
            first_momentum (float, optional): 一阶动量系数（例如 Adam 中的 beta1）。默认为 None。
            second_momentum (float, optional): 二阶动量系数（例如 Adam 中的 beta2）。默认为 None。
            variant (str): 基础优化器类型，如 'sgd', 'adam'。默认为 'sgd'。
            target_group_sparsity (float): 目标参数组稀疏度。范围在 0 到 1 之间，表示希望剪掉的参数组的比例。默认为 0.5。
            start_pruning_step (int): 开始进行剪枝操作的训练步数。默认为 0。
            pruning_steps (int): 在一个剪枝周期内，执行剪枝操作的步数。默认为 1。
            pruning_periods (int): 剪枝操作的周期数。默认为 1。
            device (str): 计算设备，如 'cuda' 或 'cpu'。默认为 'cuda'。
            dampening (float, optional): 用于 SGD 的 dampening 参数。默认为 None。
            group_divisible (int): 确保剪枝后的参数组数量是此值的倍数。默认为 1。
            fixed_zero_groups (bool): 是否在剪枝后固定那些已经被置零的参数组，使其不再更新。默认为 True。
            importance_score_criteria (str): 参数组重要性评分标准。'default' 可能表示使用 HESSO 默认的基于 Hessian 的方法。默认为 'default'。
        返回：
            HESSO: 初始化后的 HESSO 优化器实例。
        """
        from .optimizer import HESSO  # 动态导入 HESSO 优化器类，避免循环导入或在不需要时加载
        # 创建 HESSO 优化器实例
        self._optimizer = HESSO(
            # params: 从 OTO 的图对象 (_graph) 中获取参数组。
            # _graph.get_param_groups() 应该返回一个适合优化器处理的参数列表或参数组列表。
            params=self._graph.get_param_groups(),
            lr=lr,                                                  # 学习率
            weight_decay=weight_decay,                              # 权重衰减（L2 正则化）
            first_momentum=first_momentum,                          # 一阶动量系数
            second_momentum=second_momentum,                        # 二阶动量系数
            dampening=dampening,                                    # dampening 参数（用于 SGD）
            variant=variant,                                        # 基础优化器类型
            target_group_sparsity=target_group_sparsity,            # 目标组稀疏度
			start_pruning_step=start_pruning_step,                  # 开始剪枝的训练步数
            pruning_periods=pruning_periods,                        # 剪枝周期数
            pruning_steps=pruning_steps,                            # 每个周期的剪枝步数
            group_divisible=group_divisible,                        # 组可除性约束
            importance_score_criteria=importance_score_criteria,    # 重要性评分标准
            device=device,                                          # 计算设备
            alpha_scheduler_cfg=alpha_scheduler_cfg,                # Alpha 调度器配置
            seed=seed,                                              # 随机种子       
        )

        return self._optimizer                                      # 返回创建的优化器实例

   
    def hessocric(self, variant='sgd', lr=0.01, first_momentum=None, second_momentum=None, dampening=None, weight_decay=None, \
                target_group_sparsity=0.5, tolerance=0, group_divisible=1, \
                start_cric_step=0, max_cycle_period=10, sampling_steps=None, hybrid_training_steps=None, \
                importance_score_criteria='default'):
    
        from .optimizer import HESSOCRIC  # 动态导入 HESSOCric 优化器类

        self.optimizer = HESSOCRIC(params=self._graph.get_param_groups(),                  # 优化器的参数组
                                    variant=variant,                                        # 基础优化器类型
                                    lr=lr,                                                  # 学习率
                                    first_momentum=first_momentum,                          # 一阶动量系数
                                    second_momentum=second_momentum,                        # 二阶动量系数
                                    dampening=dampening,                                    # dampening 参数（用于 SGD）
                                    weight_decay=weight_decay,                              # 权重衰减（L2 正则化）
                                    target_group_sparsity=target_group_sparsity,            # 目标组稀疏度
                                    tolerance=tolerance, 
                                    group_divisible=group_divisible,
                                    start_cric_step=start_cric_step, 
                                    max_cycle_period=max_cycle_period, 
                                    sampling_steps=sampling_steps, 
                                    hybrid_training_steps=hybrid_training_steps,
                                    importance_score_criteria=importance_score_criteria,    # 重要性评分标准
                                    )
        return self.optimizer                                      # 返回创建的优化器实例
    

    def skip_operators(self, operator_list=list()):
        """
        标记某些算子为“跳过”，不参与剪枝或压缩过程。
        参数：
            operator_list: 要跳过的算子名称列表。
        """
        self._graph.skip_operators(operator_list)

    def set_trainable(self):
        """
        根据剪枝依赖图设置哪些参数为可训练状态。
        """
        self._graph.set_trainable()

    def construct_subnet(self, merge_lora_to_base=False, unmerge_lora_to_base=False, export_huggingface_format=False, export_float16=False, out_dir='./', \
                 full_group_sparse_model_dir=None, compressed_model_dir=None, save_full_group_sparse_model=True, ckpt_format='torch'):
        """
        根据剪枝图构建最终子网络并保存模型。
        参数：
            merge_lora_to_base: 是否合并 LoRA 权重到基础模型。
            unmerge_lora_to_base: 是否还原 LoRA 修改。
            export_huggingface_format: 是否导出为 HuggingFace 格式。
            export_float16: 是否使用 float16 精度保存。
            out_dir: 默认输出目录。
            full_group_sparse_model_dir: 全组稀疏模型保存路径。
            compressed_model_dir: 压缩模型保存路径。
            save_full_group_sparse_model: 是否保存全组稀疏模型。
            ckpt_format: 检查点保存格式（如 torch、safetensors 等）。
        """
        # 确定全组稀疏模型的保存目录。
        # 如果 full_group_sparse_model_dir 未指定（为 None），则使用默认的 out_dir。否则，使用用户指定的 full_group_sparse_model_dir。
        full_group_sparse_model_dir = out_dir if full_group_sparse_model_dir is None else full_group_sparse_model_dir
        
        # 确定全组稀疏模型的保存目录。
        # 如果 compressed_model_dir 未指定（为 None），则使用默认的 out_dir。否则，使用用户指定的 compressed_model_dir。
        compressed_model_dir = out_dir if compressed_model_dir is None else compressed_model_dir
        
        # 根据当前的压缩模式 (self._mode) 执行不同的操作。
        if self._mode == 'prune':
            # 如果是剪枝模式 ('prune')：
            # 调用 automated_pruning_compression 函数执行自动剪枝和压缩。该函数会返回压缩后模型的路径和全组稀疏模型的路径。
            self.compressed_model_path, self.full_group_sparse_model_path = automated_pruning_compression(
                oto_graph=self._graph,                                          # 传入 OTO 内部维护的图结构。
                model=self._model,                                              # 传入原始模型。
                merge_lora_to_base=merge_lora_to_base,                          # 是否合并 LoRA 权重到基础模型。
                unmerge_lora_to_base=unmerge_lora_to_base,                      # 是否从基础模型中移除/还原 LoRA 修改。
                export_huggingface_format=export_huggingface_format,            # 是否导出为 HuggingFace Transformers 库兼容的格式。
                export_float16=export_float16,                                  # 是否将模型权重导出为 float16 精度。
                full_group_sparse_model_dir=full_group_sparse_model_dir,        # 指定全组稀疏模型的保存目录。
                compressed_model_dir=compressed_model_dir,                      # 指定最终压缩模型的保存目录。
                save_full_group_sparse_model=save_full_group_sparse_model,      # 是否保存全组稀疏模型（通常是剪枝后但未移除零权重结构的模型）。
                ckpt_format=ckpt_format                                         # 指定检查点（模型权重）的保存格式，如 'torch' 或 'safetensors'。
            )
        elif self._mode == 'erase':
            # 如果是擦除模式 ('erase')：
            # 当前版本尚未实现此功能，因此抛出 NotImplementedError 异常。
            raise NotImplementedError

    def random_set_zero_groups(self, target_group_sparsity=None):
        """
        随机设置一部分参数组为零，达到指定的目标稀疏度。
        参数：
            target_group_sparsity: 目标稀疏度（0~1）。
        """
        self._graph.random_set_zero_groups(target_group_sparsity=target_group_sparsity)

    def mark_unprunable_by_node_ids(self, node_ids=list()):
        """
        根据节点 ID 标记不可剪枝的节点组。
        参数：
            node_ids: 不可剪枝节点 ID 列表。
        """
        # 遍历图对象中所有的节点组 (node_groups)。
        for node_group in self._graph.node_groups.values():
            # 对于每个节点组，再遍历传入的 node_ids 列表。
            for node_id in node_ids:
                # 检查当前的 node_id 是否存在于当前 node_group 的节点集合 (node_group.nodes) 中。
                if node_id in node_group.nodes:
                    # 如果找到了匹配的节点 ID，则将该节点组的 is_prunable 属性设置为 False。
                    # 这意味着这个节点组将不会被后续的剪枝过程考虑。
                    node_group.is_prunable = False

    def mark_unprunable_by_param_names(self, param_names=list()):
        """
        根据参数名标记不可剪枝的节点组。
        参数：
            param_names: 不可剪枝参数名列表。
        """
        param_names_set = set(param_names)
        for node_group in self._graph.node_groups.values():
            if set(node_group.param_names) & param_names_set:
                node_group.is_prunable = False

    def compute_flops(self, in_million=True, in_billion=False):
        """
        计算模型 FLOPs（浮点运算量）。
        参数：
            in_million: 是否以百万单位返回。
            in_billion: 是否以十亿单位返回。
        """
        return self._graph.compute_flops(in_million=in_million, in_billion=in_billion)

    def compute_num_params(self, in_million=True, in_billion=False):
        """
        计算模型参数总量。
        参数：
            in_million: 是否以百万单位返回。
            in_billion: 是否以十亿单位返回。
        """
        return self._graph.compute_num_params(in_million=in_million, in_billion=in_billion)
    