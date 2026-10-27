import os, sys, torch
sys.path.insert(0, '/workspace/OTOv2_v2')
from only_train_once import OTO
from model.model import resnet110
from Adaptive_Fusion_Temporal_Drift_Loss_Spike import load_checkpoint_if_exists

device = 'cpu'
ckpt = '/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model_snapshots/chkpt_epoch_60.pth'
model = resnet110(num_classes=100)
load_checkpoint_if_exists(model, ckpt, device)

dummy_input = torch.rand(1, 3, 32, 32)
oto = OTO(model=model, dummy_input=dummy_input)

optimizer_prune = oto.hesso(variant='sgd', lr=0.01, target_group_sparsity=0.3)
optimizer_prune.compute_importance_scores()

prunable = [g for g in optimizer_prune.param_groups if g.get('is_prunable') and not g.get('is_auxiliary')]
print("Prunable Groups:", len(prunable))

for i, g in enumerate(prunable[:5]):
    l1 = g['importance_scores'].get('l1_avg_magnitude')
    print(f"Group {i} len {g['num_groups']} L1: {l1}")

sum_l1 = sum(float((g['importance_scores'].get('l1_avg_magnitude', torch.tensor(0)).sum())) for g in prunable)
print("Total L1 sum:", sum_l1)
