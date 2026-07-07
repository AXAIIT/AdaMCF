'''
utils.py 文件包含用于 PyTorch ONNX 导出的辅助函数。
主要功能包括：
图优化:
_optimize_trace_graph_no_onnx_operator: 这个函数应用了一系列的 PyTorch JIT (Just-In-Time) passes 来优化计算图，为 ONNX 导出做准备。它执行诸如内联、常量传播、死代码消除 (DCE)、算子融合等操作。
_split_tensor_list_constants: 将图中的常量张量列表节点拆分成单独的常量节点和 prim::ListConstruct 节点，这有助于 ONNX 转换。
_is_constant_tensor_list: 检查一个节点是否是常量张量列表。
字符串处理:
_get_str_inside_parenthesis: 从带有括号的字符串中提取括号内的内容。
_get_tensor_shape: 从节点的字符串表示中解析张量的形状。
数值缩放:
_scale_value: 将数值按百万 (MILLION) 或十亿 (BILLION) 进行缩放。
这个文件提供了在将 PyTorch 模型导出为 ONNX 格式过程中所需的图处理、优化和数据解析工具。

utils.py 文件中的函数以 _ 开头，意味着它们是设计为供 only_train_once.graph 包内部使用的辅助函数，而不是供外部代码直接调用的公共接口。
'''
from torch import _C                    # PyTorch C++ 后端接口
import torch._C._onnx as _C_onnx        # PyTorch ONNX C++ 后端接口
from torch.onnx import (
    symbolic_helper                     # ONNX 导出的符号辅助函数
)
import textwrap                         # 用于处理文本换行
from torch.onnx._globals import GLOBALS # ONNX 导出的全局配置

def _is_constant_tensor_list(node):
    """
    检查给定的 JIT 节点是否是一个常量张量列表或可选张量列表。
    Args:
        node (_C.Node): 要检查的 JIT 节点。
    Returns:
        bool: 如果节点是常量张量列表或可选张量列表，则返回 True，否则返回 False。
    """
    # 检查节点类型是否为 prim::Constant
    if node.kind() != "prim::Constant":
        return False
    output_type = node.output().type()
    # 检查输出类型是否为张量列表
    if output_type.isSubtypeOf(_C.ListType.ofTensors()):
        return True
    # 检查输出类型是否为可选张量列表
    if output_type.isSubtypeOf(_C.ListType(_C.OptionalType.ofTensor())):
        return True
    # 如果以上都不是，则返回 False

def _split_tensor_list_constants(g, block):
    """
    递归地遍历计算图块 (block)，将常量张量列表节点拆分成单独的常量节点
    和一个 prim::ListConstruct 节点。这有助于 ONNX 转换，因为 ONNX 可能
    不直接支持包含整个列表的单个常量节点。
    Args:
        g (_C.Graph): PyTorch JIT 计算图。
        block (_C.Block): 当前正在处理的计算图块。
    """
    # 遍历当前块中的所有节点
    for node in block.nodes():
        # 递归处理节点内的子块
        for subblock in node.blocks():
            _split_tensor_list_constants(g, subblock)
        # 检查当前节点是否是常量张量列表
        if _is_constant_tensor_list(node):
            inputs = []
            # 获取常量列表中的每个张量值
            for val in node.output().toIValue():
                # 为每个张量值创建一个新的常量节点
                input = g.insertConstant(val)
                # 将新常量节点移动到原始列表常量节点之前
                input.node().moveBefore(node)
                # 复制元数据（例如源代码位置）
                input.node().copyMetadata(node)
                inputs.append(input)

            # 创建一个新的 prim::ListConstruct 节点，使用上面创建的单个常量节点作为输入
            lc = (
                g.create("prim::ListConstruct", inputs) # 创建 ListConstruct 节点
                .insertBefore(node)                     # 插入到原始节点之前
                .output()                               # 获取输出值
                .setType(_C.ListType.ofTensors())       # 设置输出类型为张量列表
            )
            # 复制元数据到新的 ListConstruct 节点
            lc.node().copyMetadata(node)
            # 将原始常量列表节点的所有用途替换为新的 ListConstruct 节点的输出
            node.output().replaceAllUsesWith(lc)
            # 此时，原始的常量列表节点 'node' 变成无用节点，后续的 DCE pass 会移除它

def _optimize_trace_graph_no_onnx_operator(
    graph: _C.Graph,
    operator_export_type: _C_onnx.OperatorExportTypes,
    _disable_torch_constant_prop: bool = False,
    fixed_batch_size: bool = False,
    params_dict=None,
    dynamic_axes=None,
    input_names=None,
    module=None,
):
    """
    应用一系列 PyTorch JIT passes 来优化追踪到的计算图，为 ONNX 导出做准备。
    这个函数主要执行标准的 JIT 优化，但不包括特定于 ONNX 算子的转换（这些转换通常在 `_C._jit_pass_onnx` 中完成，这里被注释掉了或在外部调用）。
    Args:
        graph (_C.Graph): 要优化的 PyTorch JIT 计算图。
        operator_export_type (_C_onnx.OperatorExportTypes): ONNX 算子导出类型 (例如 RAW, ONNX, ATEN)。
        _disable_torch_constant_prop (bool, optional): 是否禁用 PyTorch 的常量传播优化。默认为 False。
        fixed_batch_size (bool, optional): 是否假设固定的批处理大小。默认为 False。
        params_dict (dict, optional): 包含模型参数的字典。默认为 None。
        dynamic_axes (dict, optional): 指定输入的动态轴信息。默认为 None。
        input_names (list, optional): 输入节点的名称列表。默认为 None。
        module (torch.nn.Module, optional): 原始的 PyTorch 模型实例。默认为 None。
    Returns:
        _C.Graph: 优化后的 PyTorch JIT 计算图。
    """
    # 检查模型参数的字典是否为空，如果为空则初始化为空字典
    if params_dict is None:
        params_dict = {}

    # === 基础优化和清理 ===
    # 内联所有函数调用，将子图合并到主图中
    _C._jit_pass_inline(graph)

    # 移除并发相关的 fork/wait 节点，fork 可以启动一个新的执行线程（逻辑上的），而 wait 则等待该线程完成。
    _C._jit_pass_inline_fork_wait(graph)
    # 运行 Lint pass 检查图的有效性
    _C._jit_pass_lint(graph)
    # 处理 Autograd Function 节点，torch.autograd.Function 允许用户自定义新的自动求导操作。
    # _C._jit_pass_onnx_autograd_function_process 这个优化遍专门用于处理这些代表 autograd.Function 的节点，以便它们能够被正确地转换为 ONNX 兼容的表示。
    _C._jit_pass_onnx_autograd_function_process(graph)
    # 展开所有元组 (tuple) 结构，扁平化数据流。
    _C._jit_pass_lower_all_tuples(graph)

    # === 常量传播和死代码消除 ===
    # PyTorch 现在会将 ones/zeros 等操作记录到追踪图中，之前是常量。
    # 使用常量传播来维持当前的 ONNX 支持水平，而无需为所有这些操作实现符号导出。
    if _disable_torch_constant_prop is False:
        _C._jit_pass_constant_propagation(graph)

    # 将常量张量列表拆分为单独的常量和 ListConstruct
    _split_tensor_list_constants(graph, graph)
    # 运行死代码消除 (DCE)，移除图中不再需要的节点和值，这可以清除 symbolic_override 等操作留下的无用部分
    _C._jit_pass_dce(graph)
    _C._jit_pass_lint(graph)

    # === 融合和模式重写 ===
    # 公共子表达式消除 (CSE) 可以在禁用缓存的 Autocast 下提高性能
    # (参考: https://github.com/pytorch/pytorch/issues/84092)
    # 必须在 _C._jit_pass_erase_number_types 之前运行，以防止类型替换问题
    # 尝试执行公共子表达式消除 (CSE) 优化。
    # _C._jit_pass_cse 会遍历计算图 graph，查找并消除重复的计算（即公共子表达式）。
    # 如果 CSE 成功地对图进行了一些更改（即找到了并消除了某些公共子表达式），
    # 这个函数调用通常会返回 True。如果图没有发生任何变化（没有找到可消除的 CSE），
    # 则可能返回 False。
    if _C._jit_pass_cse(graph):
        _C._jit_pass_onnx_lint(graph) # 专门针对 ONNX 导出的要求和约束来检查图。

    # 标准化图融合器 (Graph Fuser) 创建的操作
    _C._jit_pass_canonicalize_graph_fuser_ops(graph)
    _C._jit_pass_lint(graph)
    # 应用 Peephole 优化（基于小模式的局部优化），允许不安全的优化
    _C._jit_pass_peephole(graph, True)
    # 融合 Add 和 MatMul (addmm) 操作
    _C._jit_pass_fuse_addmm(graph)
    _C._jit_pass_lint(graph)

    # 再次应用 Peephole 优化和元组展开
    _C._jit_pass_peephole(graph, True)
    _C._jit_pass_lower_all_tuples(graph)

    # === ONNX 特定预处理 ===
    # 在 _jit_pass_onnx 中，会为每个节点调用符号函数进行转换。

    '''
    # 作用: 移除或替换计算图中的原地 (in-place) 操作，为 ONNX 导出做准备。
    # 详细解释:
    #   - 原地操作: 指那些直接修改输入张量内容的 PyTorch 操作，例如 tensor.add_(other_tensor) 中的下划线版本。
    #               这些操作在 PyTorch 中可以节省内存，因为它们避免了为输出创建新的张量。
    #   - ONNX 的限制: ONNX 格式通常期望算子是纯函数式的，即它们不修改输入，而是产生新的输出。
    #                  直接导出包含原地操作的图到 ONNX 可能会导致问题或不兼容。
    #   - 此 pass 的行为:
    #     1. 识别图中的原地操作。
    #     2. 尝试将这些原地操作替换为其对应的非原地 (out-of-place) 版本。
    #        例如，aten::add_ (原地加法) 可能会被替换为 aten::add (非原地加法)。
    #        这意味着会创建一个新的张量来存储操作的结果，而不是修改原始输入。
    #     3. 在某些情况下，如果无法简单替换（例如，原地操作的特定语义难以用非原地操作精确模拟，
    #        或者该原地操作没有直接的非原地对应），此 pass 可能需要更复杂的转换，
    #        或者如果无法安全转换，可能会发出警告或错误。
    #   - module 参数: 传入原始的 PyTorch 模型实例 (module) 可能有助于此 pass 理解某些操作的上下文，
    #                 或者在需要访问模型状态（如缓冲区）来进行转换时提供信息。
    #   - 目的: 确保图中的所有操作都符合 ONNX 对算子行为（非原地性）的要求，从而实现成功的 ONNX 导出。
    '''
    _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)    # 移除或替换原地操作

    '''
    # 作用: 执行一系列针对 ONNX 导出的通用预处理步骤。
    # 详细解释:
    #   - 通用预处理: 这个 pass 通常包含一些在将 PyTorch 操作转换为 ONNX 算子之前需要进行的准备工作。
    #                 它不是针对某一个特定类型的操作，而是对图进行更广泛的调整。
    #   - 可能的操作包括:
    #     1. 节点规范化: 将某些 PyTorch 特有的节点或模式转换为更通用或更接近 ONNX 期望的形式。
    #     2. 属性调整: 修改某些节点的属性，使其符合 ONNX 算子的规范。
    #     3. 插入辅助节点: 在必要时插入一些辅助节点（例如类型转换、形状调整节点），以确保数据流和类型
    #                      在转换为 ONNX 后仍然正确。
    #     4. 处理特殊情况: 解决一些已知的 PyTorch JIT 图与 ONNX 导出之间可能存在的细微不兼容问题。
    #   - 上下文依赖: 这个 pass 的具体行为可能依赖于 `operator_export_type` (例如，是导出到标准的 ONNX，
    #                 还是用于 Caffe2 的 ATEN 后备模式等)，以及当前的 ONNX opset 版本。
    #   - 目的: 在主要的 ONNX 转换步骤 (_C._jit_pass_onnx) 之前，对图进行必要的清理和准备，
    #           使得后续的符号函数 (symbolic functions，负责将每个 PyTorch 操作转换为 ONNX 操作)
    #           能够更容易、更可靠地工作。它可以被看作是为更复杂的 ONNX 转换铺平道路的一系列小型修复和调整。
    '''
    _C._jit_pass_onnx_preprocess(graph)                             # ONNX 预处理

    # ONNX 不支持元组，尝试移除它们
    _C._jit_pass_lint(graph)

    # ONNX 只支持张量运算，但 Python 中 1 / 2 = 0.5，而 tensor(1) / tensor(2) = 0 (整数除法)，此 pass 准备除法操作以符合 ONNX 预期
    _C._jit_pass_prepare_division_for_onnx(graph)

    # 移除图中的 Print 操作
    _C._jit_pass_onnx_remove_print(graph)
    # Caffe2 特定的 ONNX 预处理
    _C._jit_pass_onnx_preprocess_caffe2(graph)

    # === 量化处理 (如果需要) ===
    # 清理之前的量化操作记录
    symbolic_helper._quantized_ops.clear() 
    # 为卷积和线性操作解包量化权重，并插入到图中
    _C._jit_pass_onnx_unpack_quantized_weights(
        graph, params_dict, symbolic_helper.is_caffe2_aten_fallback()
    )
    # 如果使用了 Caffe2 ATen 后备模式
    if symbolic_helper.is_caffe2_aten_fallback():
        # 在每个卷积操作前后插入 permute 操作以确保正确的通道顺序 (NHWC vs NCHW)
        _C._jit_pass_onnx_quantization_insert_permutes(graph, params_dict)
        # 查找并移除连续的、实际上是无操作 (no-op) 的 permute 对 (例如 NHWC -> NCHW -> NHWC)
        _C._jit_pass_custom_pattern_based_rewrite_graph(
            textwrap.dedent( # 定义要查找的模式
                """\
                graph(%Pi):
                    %Pq = quantized::nhwc2nchw(%Pi)
                    %Pr = quantized::nchw2nhwc(%Pq)
                    return (%Pr)"""
            ),
            textwrap.dedent( # 定义替换后的模式
                """\
                graph(%Ri):
                    return (%Ri)"""
            ),
            graph,
        )

    # === 类型擦除和形状推断 ===
    # ONNX 只支持张量，将所有数值类型 (int, float) 转换为张量
    _C._jit_pass_erase_number_types(graph)
    # 如果启用了 ONNX 形状推断
    if GLOBALS.onnx_shape_inference:
        input_names = [] if input_names is None else input_names
        dynamic_axes = {} if dynamic_axes is None else dynamic_axes
        # 设置输入的动态形状信息
        _C._jit_pass_onnx_set_dynamic_input_shape(graph, dynamic_axes, input_names)
    _C._jit_pass_onnx_lint(graph)

    # === 主要的 ONNX 转换 Pass (通常在这里调用，但此函数可能跳过) ===
    graph = _C._jit_pass_onnx(graph, operator_export_type) # 将 PyTorch 操作转换为 ONNX 操作
    # except:
    #     pass # 错误处理（可能被省略）
    _C._jit_pass_onnx_lint(graph)
    _C._jit_pass_lint(graph)

    # === ONNX 后处理和最终优化 ===
    # 进行 ONNX 标量类型分析
    _C._jit_pass_onnx_scalar_type_analysis(
        graph, True, GLOBALS.export_onnx_opset_version
    )
    _C._jit_pass_lint(graph)

    # 应用 ONNX 特定的 Peephole 优化
    _C._jit_pass_onnx_peephole(
        graph, GLOBALS.export_onnx_opset_version, fixed_batch_size
    )
    _C._jit_pass_lint(graph)

    # 此时图不再是有效的 JIT 图，因为类型已被替换（例如 int -> Tensor），
    # 它包含了一些实际不存在的操作符。不能运行普通的 DCE，因为它会尝试查找
    # 操作符是否有副作用而失败。但可以运行一个不需要查找副作用的 DCE 变体。
    _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    _C._jit_pass_lint(graph)
    # 标准化图表示
    graph = _C._jit_pass_canonicalize(graph)
    _C._jit_pass_lint(graph)
    # 尝试进行 ONNX 图级别的形状和类型推断
    try:
        if GLOBALS.onnx_shape_inference:
            _C._jit_pass_onnx_graph_shape_type_inference(
                graph, params_dict, GLOBALS.export_onnx_opset_version
            )
    except: # 捕获可能的推断错误
        pass
    # 返回最终优化后的图
    return graph

def _get_str_inside_parenthesis(str_to_processed, prefix_str=None):
    """
    从带有括号的字符串中提取第一个顶级括号对内的内容。
    例如，对于 "Float(1, 2, 3)" 和 prefix_str="Float"，它将返回 "1, 2, 3"。
    Args:
        str_to_processed (str): 要处理的输入字符串。
        prefix_str (str, optional): 期望字符串具有的前缀。如果提供，只有当字符串以此前缀开头时才进行处理。默认为 None。
    Returns:
        str or None: 第一个顶级括号内的内容字符串，如果找不到匹配的括号或前缀不匹配，则返回 None。
    """
    # 如果提供了前缀字符串，则检查输入字符串是否以该前缀开头
    if not str_to_processed.startswith(prefix_str):
        return None
    
    # 初始化一个空列表作为栈，用于跟踪括号的配对情况。遇到 '(' 则入栈，遇到 ')' 则出栈。
    stack = []
    # 计算截取子字符串的起始索引。
    # 假设 prefix_str 后面紧跟着一个 '('，我们想提取 '(' 之后的内容。
    # 例如，如果 prefix_str = "graph"，str_to_processed = "graph(content)"
    # len(prefix_str) = 5。'(' 的索引是 5。我们想从索引 6 开始提取 "content"。
    start_idx = len(prefix_str) + 1
    # 初始化结束索引为 -1。这个变量将在循环中递增，表示当前字符的索引。
    end_idx = -1 
    # 遍历输入字符串中的每个字符
    for c in str_to_processed:
        # 如果当前字符是左括号 '('
        if c == '(':
            # 将左括号压入栈中
            stack.append(c)
        # 如果当前字符是右括号 ')'
        elif c == ')':
            # 从栈中弹出一个元素（预期的行为是弹出一个左括号）
            # 这假设了括号是基本配对正确的，否则 stack.pop() 可能在空栈上调用导致错误。
            stack.pop()
        # 更新 end_idx，使其指向当前处理的字符 c 的索引
        end_idx += 1
        # 检查栈是否为空，并且当前的结束索引已经超过了前缀字符串的长度
        # len(stack) == 0 表示我们找到了一个与初始左括号配对的右括号，形成了一个完整的括号对。
        # end_idx > len(prefix_str) 确保这个闭合的括号是在 prefix_str 之后的部分，
        # 并且至少是在紧随 prefix_str 的那个 '(' 之后。
        if len(stack) == 0 and end_idx > len(prefix_str):
            # 如果条件满足，说明已经找到了第一个顶级括号对的末尾，跳出循环。
            break
    # 返回从start_idx到end_idx（不包括end_idx）的子字符串。这正是第一个顶级括号对内部的内容。
    return str_to_processed[start_idx : end_idx]

def _get_tensor_shape(str_to_processed, prefix_str='Float'):
    """
    从节点的字符串表示中解析张量的形状。
    假设形状信息在第一个括号内，由逗号分隔的数字组成。
    例如："Float(1, 3, 224, 224)" -> [1, 3, 224, 224]
    Args:
        str_to_processed (str): 包含张量信息的节点字符串。
        prefix_str (str, optional): 期望字符串具有的前缀（通常是数据类型）。默认为 'Float'。
    Returns:
        list[int] or None: 解析出的形状列表，如果无法解析则返回 None。
    """
    # 提取括号内的内容，例如对于 "Float(1, 2, 3)" 和 prefix_str="Float"，output_str 会是 "1, 2, 3"
    output_str = _get_str_inside_parenthesis(str_to_processed, prefix_str=prefix_str)
    # 如果 _get_str_inside_parenthesis 未能成功提取内容（例如，前缀不匹配或没有找到括号），则返回 None
    if output_str is None:
        return None
    # 按逗号分割从括号内提取出来的字符串。例如，如果 output_str 是 "1, 2, 3"，则 output_str_splits 会是 ["1", " 2", " 3"]
    output_str_splits = output_str.split(',')
    # 初始化一个空列表，用于存储解析出来的形状维度
    output_shapes = []
    # 遍历分割后的字符串列表中的每一项
    for item in output_str_splits:
        # 去除当前项两端的空白字符（例如，" 2" -> "2"）
        item = item.strip() 
        # 检查处理后的字符串项是否只包含数字字符
        if item.isnumeric():
            # 如果是数字，则将其转换为整数并添加到 output_shapes 列表中
            output_shapes.append(int(item))
        else:
            # 如果遇到任何非数字的项（例如，可能是 "1, 2, requires_grad=True" 中的 "requires_grad=True"），
            # 则立即停止解析，因为假设形状信息只包含数字，并且是连续的。
            # 这意味着只提取开头连续的数字作为形状。
            break
    # 在遍历和解析完成后，返回包含所有成功解析出的形状维度的列表。
    # 如果 output_str_splits 为空（例如，括号内为空 "Float()"），或者所有项都不是数字，
    # 则 output_shapes 将是一个空列表。# 如果解析过程中因为非数字项而中断，则返回已解析的部分。
    return output_shapes

# 定义常量用于数值缩放
MILLION = 1e6 # 一百万
BILLION = 1e9 # 十亿

def _scale_value(value, in_million=True, in_billion=False):
    """
    将数值按百万或十亿进行缩放。
    Args:
        value (float or int): 要缩放的数值。
        in_million (bool, optional): 是否除以一百万。默认为 True。
        in_billion (bool, optional): 是否除以十亿。如果 in_million 为 True，则此参数无效。
                                     默认为 False。
    Returns:
        float: 缩放后的数值。
    """
    if in_million:
        # 除以一百万
        value /= float(MILLION)
    elif in_billion:
        # 除以十亿
        value /= float(BILLION)
    return value
