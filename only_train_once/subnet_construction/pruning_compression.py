'''
pruning_compression.py 文件定义了一个名为 automated_pruning_compression 的函数，其主要功能是自动化地对 PyTorch 模型进行剪枝和压缩。
具体作用包括：
模型准备:
根据 ckpt_format (torch 或 onnx) 和 export_huggingface_format 确定保存模型的文件名和路径。
创建保存完整稀疏模型和压缩模型的目录。
如果 export_float16 为 True，将模型转换为半精度 (float16)。
保存完整组稀疏模型 (可选):
如果 save_full_group_sparse_model 为 True，在进行压缩之前，根据指定的格式 (Hugging Face, torch, onnx) 保存当前状态的模型（可能已经应用了某种稀疏化，但未移除参数）。
计算冗余索引:
调用 oto_graph.set_pruning_redundant_idxes() 来计算图中可以被剪枝的冗余参数索引。oto_graph 似乎是一个表示模型计算图和依赖关系的对象。
执行剪枝:
第一遍 (输出维度剪枝): 遍历图中的节点组 (node_group)，对可剪枝的组执行输出维度剪枝 (prune_out_dim)。
第二遍 (输入维度剪枝): 再次遍历节点组和节点，根据其前驱节点组的输出维度剪枝情况，计算并执行输入维度剪枝 (prune_in_dim)。这需要查找依赖关系，并处理特殊情况（如多头注意力、Flatten+Linear 结构）。
LoRA 处理 (可选):
如果 merge_lora_to_base 为 True，将 LoRA 权重合并到基础模型中。
如果 unmerge_lora_to_base 为 True，将 LoRA 权重从基础模型中分离（这通常在剪枝前完成，但代码中放在剪枝后，可能用于特定目的或后续处理）。
保存压缩模型:
将经过剪枝（参数被实际移除或变小）后的模型，根据指定的格式 (Hugging Face, torch, onnx) 保存到 compressed_model_path。
返回路径:
返回压缩后模型和（如果保存了）完整组稀疏模型的路径。
总而言之，该文件实现了一个基于 oto_graph 分析结果，自动执行模型剪枝（移除冗余的输入/输出通道/维度）并保存压缩后模型的流程。
'''
 
import torch
import os

def automated_pruning_compression(oto_graph, model, merge_lora_to_base, unmerge_lora_to_base, export_huggingface_format, export_float16, \
                          full_group_sparse_model_dir, compressed_model_dir, save_full_group_sparse_model, ckpt_format):
    """
    自动化执行模型剪枝和压缩。
    Args:
        oto_graph: OTO图对象，包含模型的计算图和节点组信息。
        model: 需要剪枝和压缩的 PyTorch 模型。
        merge_lora_to_base (bool): 是否将 LoRA 权重合并到基础模型。
        unmerge_lora_to_base (bool): 是否将 LoRA 权重从基础模型分离。
        export_huggingface_format (bool): 是否以 Hugging Face 格式保存模型。
        export_float16 (bool): 是否将模型转换为 float16 半精度。
        full_group_sparse_model_dir (str): 保存完整组稀疏模型的目录路径。
        compressed_model_dir (str): 保存压缩后模型的目录路径。
        save_full_group_sparse_model (bool): 是否在剪枝前保存完整组稀疏模型。
        ckpt_format (str): 保存模型的格式 ('torch' 或 'onnx')。
    Returns:
        tuple: 包含压缩后模型路径和完整组稀疏模型路径的元组。
    """
    
    
    full_group_spase_model_name = None      # 完整组稀疏模型名，执行最终的剪枝压缩之前保存
    compressed_model_name = None            # 存储最终“压缩后模型”的文件名。这是经过输入/输出维度剪枝，参数被实际移除或变小之后的模型。
    model_name_prefix =  (model.name if hasattr(model, 'name') else type(model).__name__)       # 获取模型名称前缀，优先使用 model.name，否则使用类名
    
    # 根据 ckpt_format 确定模型文件名
    if ckpt_format == 'torch':
        full_group_spase_model_name = model_name_prefix + "_full_group_sparse.pt"
        compressed_model_name = model_name_prefix + "_compressed.pt"
    elif ckpt_format == 'onnx':
        full_group_spase_model_name = model_name_prefix + "_full_group_sparse.onnx"
        compressed_model_name = model_name_prefix + "_compressed.onnx"
        
    # 构造完整的模型保存路径
    full_group_sparse_model_path = os.path.join(full_group_sparse_model_dir, full_group_spase_model_name)
    compressed_model_path = os.path.join(compressed_model_dir, compressed_model_name)

    # 如果指定导出为 Hugging Face 格式，则调整目录和路径
    if export_huggingface_format:
        # 在指定目录下创建子目录用于存放 Hugging Face 格式的模型
        full_group_sparse_model_dir = os.path.join(full_group_sparse_model_dir, 'huggingface_format_full')
        compressed_model_dir = os.path.join(compressed_model_dir, 'huggingface_format_compressed')
        # Hugging Face 格式保存的是目录，所以路径直接指向目录
        full_group_sparse_model_path = full_group_sparse_model_dir
        compressed_model_path = compressed_model_dir
        
    # 创建保存模型的目录，如果目录已存在则忽略
    os.makedirs(full_group_sparse_model_dir, exist_ok=True)
    os.makedirs(compressed_model_dir, exist_ok=True)
    
    # 如果指定导出为 float16，则将模型转换为半精度
    if export_float16:
        model.half()
    
    # 如果指定保存完整组稀疏模型
    if save_full_group_sparse_model:
        # 根据指定的格式保存模型
        if export_huggingface_format:
            model.save_pretrained(full_group_sparse_model_path)     # 使用 Hugging Face 的 save_pretrained 方法保存
        elif ckpt_format == 'torch':
            torch.save(model, full_group_sparse_model_path)   
        elif ckpt_format == 'onnx':
            # 使用 torch.onnx.export 导出为 ONNX 格式
            torch.onnx.export(
                model,
                oto_graph.dummy_input, # 需要一个虚拟输入来追踪模型结构
                full_group_sparse_model_path)
                
    oto_graph.set_pruning_redundant_idxes() # 计算并设置图中可剪枝的冗余索引

    # 第一遍剪枝：执行输出维度剪枝
    pruned_out_dim_modules = set() # 用于记录已经执行过输出维度剪枝的模块，避免重复操作
    # 遍历 OTO 图中的所有节点组
    for node_group in oto_graph.node_groups.values():
        # 如果节点组不可剪枝或只是辅助节点组，则跳过
        if not node_group.is_prunable and not node_group.is_auxiliary:
            continue
        # 对当前节点组执行输出维度剪枝，global_skip_modules 包含已剪枝的模块，防止在复杂结构中重复剪枝同一模块
        node_group.prune_out_dim(global_skip_modules=pruned_out_dim_modules)
        # 将当前节点组包含的模块添加到已剪枝集合中
        pruned_out_dim_modules = pruned_out_dim_modules.union(node_group.get_modules())

    # 第二遍剪枝：执行输入维度剪枝
    def find_incoming_node_group_stem_node(graph, node, src_ng, visited, incoming_node_groups, incoming_stem_node_ids):
        """
        递归查找影响当前节点输入的、可剪枝的源头节点组或起始节点。
        Args:
            graph: OTO图对象。
            node: 当前正在处理的节点。
            src_ng: 当前节点所属的节点组。
            visited: 记录已访问节点的字典，防止循环。
            incoming_node_groups: 用于存储找到的、影响输入的其他节点组ID的集合。
            incoming_stem_node_ids: 用于存储找到的、影响输入的起始节点ID的集合。
        """
        # 如果当前节点不属于源节点组 src_ng，说明找到了一个来自其他组的输入
        if src_ng.id not in node.node_group_ids and not src_ng.contain_node(node):
            # 将当前节点的所属组ID加入 incoming_node_groups
            incoming_node_groups.update(node.node_group_ids)
            return 
        visited[node.id] = True     # 标记当前节点为已访问
        # 遍历当前节点的直接前驱节点
        for node_in in graph.incoming(node):
            # 如果前驱节点是起始节点 (stem node)
            if node_in.is_stem():
                # 将其加入 incoming_stem_node_ids 并返回，因为找到了源头
                incoming_stem_node_ids.add(node_in)
                return     
            # 如果前驱节点未被访问过
            if not visited[node_in.id]:                    
                # 递归调用，继续向上查找
                find_incoming_node_group_stem_node(graph, node_in, src_ng, visited, incoming_node_groups, incoming_stem_node_ids)
    
    pruned_in_dim_modules = set()   # 用于记录已经执行过输入维度剪枝的模块，避免重复操作

    # 遍历 OTO 图中的所有节点组
    for node_group in oto_graph.node_groups.values():
        # 遍历当前节点组中的所有节点
        for node in node_group.nodes.values():
            # 如果当前节点的输入维度已经被剪枝过，则跳过
            if node.pruned_status['in_dim']:
                continue
            # 如果当前节点对应的模块已经被剪枝过输入维度，则跳过，这主要用于处理一个模块被多个节点共享的情况
            if node.op.module in pruned_in_dim_modules:
                continue
            # 如果当前节点的操作 (op) 没有定义 prune_in_dim 方法，则无法进行输入维度剪枝，跳过
            if not hasattr(node.op, 'prune_in_dim'):
                continue
            # 初始化用于存储影响当前 节点输入的节点组 和 起始节点的集合
            incoming_node_groups = set()
            incoming_stem_nodes = set()
            
            # 调用辅助函数查找影响 当前节点输入的源头节点组 或 起始节点
            # oto_graph.visited_dict() 用于创建新的访问记录字典
            find_incoming_node_group_stem_node(oto_graph, node, node_group, oto_graph.visited_dict(), incoming_node_groups, incoming_stem_nodes)
                
            in_dim_pruned_idxes = None      # 初始化用于输入维度剪枝的索引列表
            # 情况一：找到了影响输入的起始节点 (stem nodes)
            if len(incoming_stem_nodes) > 0:
                incoming_stem_node = next(iter(incoming_stem_nodes))                            # 取其中一个起始节点（假设只有一个主要输入源）
                incoming_ng = oto_graph.node_groups[incoming_stem_node.node_group_ids[0]]       # 获取该起始节点所属的节点组
                in_dim_pruned_idxes = incoming_ng.pruning_redundant_idxes                       # 使用该节点组的冗余索引作为输入剪枝的依据
            # 情况二：没有找到起始节点，但找到了影响输入的其他节点组
            elif len(incoming_node_groups) > 0:
                incoming_ng_id = None   # 初始化影响输入的节点组 ID
                # 遍历所有影响输入的节点组 ID
                for ng_id in incoming_node_groups:
                    ng = oto_graph.node_groups[ng_id]   # 获取该节点组
                    # 如果该节点组是可剪枝的或辅助性的，则认为它是主要的输入来源
                    if ng.is_prunable or ng.is_auxiliary:
                        incoming_ng_id = ng_id
                    # 如果该节点组不可剪枝但包含参数，说明输入维度不能随意改变，中断查找
                    elif not ng.is_prunable and len(ng.param_names) > 0:
                        incoming_ng_id = None
                        break
                # 如果没有找到合适的输入节点组，则跳过当前节点
                if incoming_ng_id is None:
                    continue # 没有找到合适的输入源，跳过当前节点
                # 获取该输入节点组
                incoming_ng = oto_graph.node_groups[incoming_ng_id]
                # 使用该节点组的冗余索引作为输入剪枝的依据
                in_dim_pruned_idxes = incoming_ng.pruning_redundant_idxes

            # 如果最终没有确定输入剪枝的索引 (in_dim_pruned_idxes is None)，则跳过当前节点
            if in_dim_pruned_idxes is None:
                continue

            # 特殊处理：针对多头注意力 (Multi-Head Attention) 等结构，检查输入节点组的操作 (op) 是否与多头注意力相关
            if hasattr(incoming_ng, 'op'):
                num_heads = 1 # 默认为1个头
                head_dim = 1  # 默认为维度1
                # 如果输入节点组的操作 (op) 属性中有 prune_mode 属性，且值为 'num_head'，说明是多头注意力
                if hasattr(incoming_ng.op, 'prune_mode') and incoming_ng.op.prune_mode == 'num_head':
                    # 如果是多头注意力，获取当前节点组的操作 (op) 的 num_heads 和 head_dim 属性
                    if hasattr(incoming_ng.op, 'num_heads'):
                        num_heads = incoming_ng.op.num_heads
                    if hasattr(incoming_ng.op, 'head_dim'):
                        head_dim = incoming_ng.op.head_dim
                    # 如果是多头且有多头维度，需要转换剪枝索引
                    if num_heads > 1 and head_dim > 1:
                        in_dim_pruned_idxes = list()    # 初始化转换后的剪枝索引列表
                        # 遍历输入节点组的冗余索引
                        for i in incoming_ng.pruning_redundant_idxes:
                            in_dim_pruned_idxes.extend([h + i * head_dim for h in range(head_dim)])
                else:
                    # 如果不是多头注意力，检查输入节点组的操作 (op) 是否有 num_heads 和 head_dim 属性
                    if hasattr(incoming_ng.op, 'num_heads'):
                        num_heads = incoming_ng.op.num_heads
                    if hasattr(incoming_ng.op, 'head_dim'):
                        head_dim = incoming_ng.op.head_dim
                    # 如果是多头且有多头维度，需要转换剪枝索引
                    if num_heads > 1 and head_dim > 1:
                        in_dim_pruned_idxes = list()    # 初始化转换后的剪枝索引列表
                        # 遍历每个头
                        for h in range(num_heads):
                            # 将每个索引转换为对应的多头维度索引
                            in_dim_pruned_idxes.extend([i + h * head_dim for i in incoming_ng.pruning_redundant_idxes]) 
                
            # 特殊处理：处理 Flatten + Linear 结构
            node_in = oto_graph.incoming(node)[0]   # 获取当前节点的直接前驱节点
            # 如果前驱是 Flatten 操作，当前是 Linear 操作
            if node_in.op_name == 'flatten' and node.op_name == 'linear':
                # 计算 Flatten 操作将每个输入组特征展平的倍数，incoming_ng.num_groups 可能是指 Flatten 前的特征分组数量（例如通道数）
                # node.op.module.in_features 是 Linear 层的总输入特征数
                # expand_time是指每个组展平后的特征数
                expand_time = node.op.module.in_features // incoming_ng.num_groups
                in_dim_pruned_idxes_refined = list()    # 初始化展平后的剪枝索引列表
                # 将基于 Flatten 前分组的剪枝索引，扩展为 Flatten 后 Linear 层的实际维度索引
                for idx in in_dim_pruned_idxes: 
                    # idx 是 Flatten 前要剪掉的组索引
                    # 将该组对应的所有展平后的维度都加入剪枝列表
                    in_dim_pruned_idxes_refined.extend([i + idx * expand_time for i in range(expand_time)])
                in_dim_pruned_idxes = in_dim_pruned_idxes_refined
            
            if not node.pruned_status['in_dim']:
                # 调用节点操作 (op) 的 prune_in_dim 方法执行实际的输入维度剪枝
                # 传入计算得到的剪枝索引和节点相关的参数名
                node.op.prune_in_dim(pruned_idxes=in_dim_pruned_idxes, param_names=node.param_names)
                # 标记当前节点的输入维度为已剪枝
                node.pruned_status['in_dim'] = True
                # 如果节点操作是基础操作 (is_basic) 且不包含 LoRA (防止重复添加共享的 LoRA 模块)，将该操作对应的模块添加到已剪枝输入维度的集合中
                if node.op.is_basic and not node_group.contain_lora():
                    pruned_in_dim_modules.add(node.op.module)

    # LoRA 处理：合并 LoRA 权重到基础模型 (通常在剪枝后进行)
    if merge_lora_to_base:
        # 检查模型是否有 merge_and_unload 方法 (通常由 PEFT 库提供)
        if hasattr(model, 'merge_and_unload'):
            # 执行合并并将 LoRA 相关层卸载
            model = model.merge_and_unload()

    # LoRA 处理：分离 LoRA 权重 (通常在剪枝前进行，这里放在剪枝后可能用于特殊目的)
    if unmerge_lora_to_base:
         # 检查模型是否有 unmerge_and_unload 方法 (通常由 PEFT 库提供)
        if hasattr(model, 'unmerge_and_unload'):
             # 执行分离
            model = model.unmerge_and_unload()
            
    # _sync_multihead_attr(model)

    # 保存最终压缩后的模型
    if export_huggingface_format:
        # 使用 Hugging Face 的 save_pretrained 方法保存
        model.save_pretrained(compressed_model_path)
    elif ckpt_format == 'torch':
        torch.save(model, compressed_model_path)
    elif ckpt_format == 'onnx':
        # 使用 torch.onnx.export 导出为 ONNX 格式
        torch.onnx.export(
            model,
            oto_graph.dummy_input, # 需要虚拟输入
            compressed_model_path)
            
    # 返回压缩后模型的路径和（如果保存了）完整组稀疏模型的路径
    return compressed_model_path, full_group_sparse_model_path
