import os
import torch
from torchvision import models

def main():
    save_dir = "/workspace/OTOv2_v2/test/pretrain_mobilenetv2_imagenet/models"
    os.makedirs(save_dir, exist_ok=True)
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1  # 官方ImageNet预训练
    model = models.mobilenet_v2(weights=weights)
    save_path = os.path.join(save_dir, "mobilenet_v2_imagenet1k_v1.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    main()