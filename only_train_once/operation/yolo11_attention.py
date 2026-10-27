"""可持久化的 YOLO11 Attention 物理子网前向适配。

原始 Ultralytics Attention 使用输入通道 ``C`` 重塑 V。结构化剪掉完整 head
或 head 内维度后，内部 V 宽度变为 ``num_heads * head_dim``，不再等于 C。
本类只把 Attention 内部宽度改为动态值；``proj`` 输出宽度仍保持 C，因此不会
改变 C2PSA 外部 residual、split 或 concat 合同。

该模块由 ``YOLO11AttentionOTO`` 在物理剪枝时惰性导入，不会让没有安装
Ultralytics 的 OTO 旧项目在导入框架时失败。
"""

from packaging.version import Version


def _load_ultralytics_attention_class():
    """按 Ultralytics 版本选择官方 Attention 定义，拒绝未知旧版布局。"""
    import ultralytics

    version = Version(str(ultralytics.__version__).split('+', 1)[0])
    if version >= Version('8.0.0'):
        from ultralytics.nn.modules.block import Attention

        return Attention, str(version), 'ultralytics.nn.modules.block.Attention'
    raise RuntimeError(
        f'YOLO11 Attention 物理剪枝不支持 ultralytics=={version}；'
        '需要具有 block.Attention(qkv/proj/pe) 的 8.x 版本。'
    )


_UltralyticsAttention, ULTRALYTICS_VERSION, ULTRALYTICS_ATTENTION_IMPORT = (
    _load_ultralytics_attention_class()
)


class OTOPrunableYOLO11Attention(_UltralyticsAttention):
    """保持外部通道不变、允许内部 head/head_dim 缩小的 YOLO11 Attention。"""

    def forward(self, x):
        batch, _, height, width = x.shape
        tokens = height * width
        inner_dim = int(self.num_heads) * int(self.head_dim)
        qkv_width = int(self.num_heads) * (
            int(self.key_dim) * 2 + int(self.head_dim)
        )
        qkv = self.qkv(x)
        if qkv.shape[1] != qkv_width:
            raise RuntimeError(
                'YOLO11 Attention QKV 物理宽度与属性合同不一致：'
                f'actual={qkv.shape[1]}, expected={qkv_width}, '
                f'num_heads={self.num_heads}, key_dim={self.key_dim}, '
                f'head_dim={self.head_dim}。'
            )
        q, k, v = qkv.reshape(
            batch,
            int(self.num_heads),
            int(self.key_dim) * 2 + int(self.head_dim),
            tokens,
        ).split([int(self.key_dim), int(self.key_dim), int(self.head_dim)], dim=2)

        attn = (q.transpose(-2, -1) @ k) * float(self.scale)
        attn = attn.softmax(dim=-1)
        value_2d = v.reshape(batch, inner_dim, height, width)
        attended = (v @ attn.transpose(-2, -1)).reshape(
            batch, inner_dim, height, width
        )
        return self.proj(attended + self.pe(value_2d))


def upgrade_yolo11_attention_module(module, prune_mode):
    """原位升级模块类型，使结构化 checkpoint 重载后仍使用动态内部宽度。"""
    required = ('qkv', 'proj', 'pe', 'num_heads', 'head_dim', 'key_dim', 'scale')
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise TypeError(f'不是受支持的 YOLO11 Attention，缺少属性：{missing}。')
    if not isinstance(module, _UltralyticsAttention):
        raise TypeError(
            'YOLO11 Attention 类与当前 Ultralytics 版本不匹配：'
            f'module={type(module)}, expected={_UltralyticsAttention}。'
        )
    if not isinstance(module, OTOPrunableYOLO11Attention):
        module.__class__ = OTOPrunableYOLO11Attention
    # 保留旧字段以便读取既有 v3/v4 checkpoint 的来源信息，但 OTO 算子不再把
    # 它当成本轮剪枝开关。新代码用 physical/active 两个字段消除二次剪枝歧义。
    module._oto_attention_prune_mode = str(prune_mode)
    module._oto_attention_physical_mode = str(prune_mode)
    module._oto_attention_active_prune_mode = 'disabled'
    module._oto_dynamic_inner_attention = True
    module._oto_ultralytics_version = ULTRALYTICS_VERSION
    module._oto_ultralytics_attention_import = ULTRALYTICS_ATTENTION_IMPORT
    return module
