"""仅封装 OTO 所需的 PyTorch ONNX 全局配置兼容入口。"""

from types import SimpleNamespace

import torch
from packaging.version import Version


TORCH_VERSION = Version(torch.__version__.split("+", 1)[0])
USES_LEGACY_ONNX_GLOBALS = TORCH_VERSION < Version("2.11.0")


def _load_onnx_globals():
    """按 PyTorch 版本加载 OTO 所需的 ONNX 全局配置对象。"""
    if USES_LEGACY_ONNX_GLOBALS:
        try:
            # 旧版本保留原生 GLOBALS，维持 OTO 原有行为。
            from torch.onnx._globals import GLOBALS
            return GLOBALS
        except ModuleNotFoundError:
            # 兼容厂商构建与非标准版本号：若实际模块缺失，安全退回最小接口。
            pass

    # PyTorch 2.11+ 已移除该私有模块；OTO 仅读取这两个字段。
    return SimpleNamespace(
        export_onnx_opset_version=16,
        onnx_shape_inference=True,
    )


GLOBALS = _load_onnx_globals()


def get_onnx_globals():
    """返回跨 OTO 模块共享的、已按版本选择的 ONNX 全局配置对象。"""
    return GLOBALS


def set_onnx_opset_version(opset_version):
    """显式设置 OTO 图追踪使用的 ONNX opset，并返回规范化后的整数值。"""
    if isinstance(opset_version, bool) or not isinstance(opset_version, int) or opset_version <= 0:
        raise ValueError(f"onnx_opset_version 必须是正整数，当前值为 {opset_version!r}。")
    get_onnx_globals().export_onnx_opset_version = opset_version
    return opset_version
