import sys, os, torch, copy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from only_train_once import OTO
from model.model import resnet110
def test_loss():
    from pretrain_resnet110_cifar100_save_epochs import get_dataloaders
    from Adaptive_Fusion_Temporal_Drift_Loss_Spike import get_loss_and_acc
    device = torch.device('cuda')
    train_loader, test_loader = get_dataloaders('/workspace/data/cifar100')
    from only_train_once.optimizer.alpha_scheduler import prepare_alpha_eval_batches
    val_batches, _ = prepare_alpha_eval_batches(trainset=train_loader.dataset, device=device, batch_size=128, num_workers=0)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    for epoch in [10, 60, 120]:
        ckpt_path = f"/workspace/OTOv2_v2/test/Ablation/resnet110_cifar100/model_snapshots/chkpt_epoch_{epoch}.pth"
        base_model = resnet110(num_classes=100).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        base_model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})
        orig_loss, _ = get_loss_and_acc(base_model, test_loader, criterion, device)
        print(f"\n--- Epoch {epoch} orig loss: {orig_loss} ---")

        for s in [0.3, 0.5, 0.8]:
            print(f" Sparsity: {s}")
            for strategy in ['L1_only', 'Fisher_only', 'Uniform_Static']:
                m = copy.deepcopy(base_model)
                oto = OTO(model=m, dummy_input=torch.rand(1,3,32,32).to(device))
                opt = oto.hesso(variant='sgd', lr=0.01, target_group_sparsity=s, device=device)
                opt.importance_score_criteria = {k:0 for k in opt._alpha_keys}
                if strategy == 'L1_only': opt.importance_score_criteria['l1_avg_magnitude'] = 1.0
                elif strategy == 'Fisher_only': opt.importance_score_criteria['fisher'] = 1.0
                elif strategy == 'Uniform_Static': 
                    for k in opt._alpha_keys: opt.importance_score_criteria[k] = 1.0 / len(opt._alpha_keys)
                m.eval()
                opt.zero_grad()
                
                # Single batch = noisy gradients, which should theoretically hurt Fisher in early epochs.
                for inputs, targets in val_batches[:1]: 
                    loss = criterion(m(inputs), targets)
                    loss.backward()
                opt.step()
                m.eval()
                ploss, _ = get_loss_and_acc(m, test_loader, criterion, device)
                print(f"  {strategy} spk: {ploss - orig_loss}")

test_loss()
