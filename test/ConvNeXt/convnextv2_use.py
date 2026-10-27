import torch
from PIL import Image
from torchvision import transforms
import timm

# 加载模型
model = timm.create_model(
    'convnextv2_huge.fcmae_ft_in22k_in1k_512',
    pretrained=False,  # 不自动下载权重
    num_classes=1000
)
# 加载本地权重

local_model_path = '/workspace/UniPruning/test/ConvNeXt/models/convnextv2_huge.fcmae_ft_in22k_in1k_512/pytorch_model.bin'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(local_model_path, map_location=device))
model.to(device)
model.eval()

# 预处理
preprocess = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 加载图片
test_image_path = '/workspace/UniPruning/data/imagenet/val/n15075141/ILSVRC2012_val_00006482.JPEG'
img = Image.open(test_image_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0).to(device)  # 增加batch维度

# 推理
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    top5_prob, top5_catid = torch.topk(probs, 5)

# 打印结果
for i in range(top5_prob.size(1)):
    print(f"Top {i+1}: 类别ID={top5_catid[0][i].item()}, 概率={top5_prob[0][i].item():.4f}")

# 可选：加载imagenet类别标签
# 可从 https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt 下载
imagenet_classes_path = '/workspace/UniPruning/test/ConvNeXt/imagenet_classes.txt'
try:
    with open(imagenet_classes_path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    for i in range(top5_prob.size(1)):
        print(f"Top {i+1}: {categories[top5_catid[0][i]]} ({top5_prob[0][i].item():.4f})")
except Exception:
    pass
