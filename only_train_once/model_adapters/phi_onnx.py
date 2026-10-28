"""原生 Phi 的无 KV 缓存 ONNX 推理包；依赖只在 ONNX 分支加载。"""
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn


class PhiLogitsForONNX(nn.Module):
    """显式生成因果 mask，避开 HF 通用 mask 构造中的不可追踪分支。

    这里只适配导出输入，不改变任何 attention、RoPE 或权重计算。4D mask
    直接交给原生 Phi，输出 logits；KV cache 和训练 loss 不属于该部署图。
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        causal = positions[:, None] >= positions[None, :]
        allowed = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()
        dtype = self.model.model.embed_tokens.weight.dtype
        additive_mask = torch.where(allowed, torch.tensor(0., device=input_ids.device, dtype=dtype),
                                    torch.tensor(torch.finfo(dtype).min, device=input_ids.device, dtype=dtype))
        return self.model(input_ids=input_ids, attention_mask=additive_mask,
                          use_cache=False, return_dict=True).logits


def export_phi_onnx(model, output):
    """导出动态 batch/sequence 图；大模型权重允许使用 ONNX external data。"""
    import onnx
    output = Path(output)
    device = next(model.parameters()).device
    tokens = torch.arange(8, dtype=torch.long, device=device).remainder(model.config.vocab_size)[None]
    mask = torch.ones_like(tokens)
    was_training = model.training
    model.eval()
    try:
        # 使用已验证环境中的 TorchScript exporter，不要求安装 onnxscript。
        torch.onnx.export(PhiLogitsForONNX(model), (tokens, mask), str(output),
            input_names=['input_ids', 'attention_mask'], output_names=['logits'],
            dynamic_axes={'input_ids': {0:'batch',1:'sequence'},
                          'attention_mask': {0:'batch',1:'sequence'},
                          'logits': {0:'batch',1:'sequence'}},
            opset_version=17, dynamo=False, external_data=True, do_constant_folding=True)
        # 以文件路径检查，支持超过 protobuf 2GB 限制的外置权重。
        onnx.checker.check_model(str(output))
    finally:
        model.train(was_training)


class PhiONNXModel:
    """CPU ONNX Runtime 推理接口；明确不提供梯度、LoRA 微调或 KV cache。"""
    inference_only = True

    def __init__(self, directory, device='cpu'):
        if str(device) != 'cpu':
            raise ValueError('当前 ONNX 加载器使用 CPUExecutionProvider，请指定 device="cpu"。')
        import onnxruntime as ort
        directory = Path(directory)
        self.plan = json.loads((directory/'architecture.json').read_text(encoding='utf-8'))
        self.config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        # 以未融合图作为数值基线。真实 Phi-2 在 ORT 默认高级融合下出现
        # 超过预设门槛的少量 logits；不放宽门槛，关闭融合后重新验证。
        # 本加载器不宣称速度最优，后续开启优化必须重新核对精度。
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.use_deterministic_compute = True
        self.session = ort.InferenceSession(str(directory/'model.onnx'), options,
                                            providers=['CPUExecutionProvider'])

    def __call__(self, input_ids, attention_mask=None):
        """返回 [batch, sequence, vocab] numpy logits，不执行生成或任务评测。"""
        ids = input_ids.detach().cpu().numpy() if torch.is_tensor(input_ids) else np.asarray(input_ids)
        if ids.ndim != 2 or ids.dtype != np.int64 or not ids.size:
            raise ValueError('input_ids 必须为非空的二维 int64。')
        if ids.min() < 0 or ids.max() >= self.config['vocab_size']:
            raise ValueError('token ID 超出词表。')
        if ids.shape[1] > self.config['max_position_embeddings']:
            raise ValueError('序列长度超过模型上下文上限。')
        if attention_mask is None:
            mask = np.ones_like(ids)
        else:
            mask = attention_mask.detach().cpu().numpy() if torch.is_tensor(attention_mask) else np.asarray(attention_mask)
            if mask.shape != ids.shape or not np.isin(mask, (0,1)).all():
                raise ValueError('attention_mask 必须与 input_ids 同形且仅包含 0/1。')
            mask = mask.astype(np.int64)
        return self.session.run(['logits'], {'input_ids':ids, 'attention_mask':mask})[0]

    def train(self, *args, **kwargs):
        raise RuntimeError('ONNX 是推理格式；继续训练或 LoRA 微调请加载 pt/safetensors。')
