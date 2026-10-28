"""兼容旧实验的导入路径；结构实现统一位于框架模型适配层。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from only_train_once.model_adapters.phi import (
    validate_native_phi, build_param_groups, group_summary, collect_pruning_plan,
    validate_plan, apply_structure, parameter_count, save_pruned_package,
    load_pruned_package,
)
from only_train_once.operation.native_phi import subset_linear
