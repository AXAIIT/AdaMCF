"""兼容旧实验的注意力导入路径；实现由框架原生 Phi 算子维护。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from only_train_once.operation.native_phi import (
    dimension_units, HeadDimPhiAttention, adapt_head_dimensions,
)
