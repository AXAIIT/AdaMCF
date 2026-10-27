import torch

def check_accuracy(model, loader):
    """
    计算模型在给定数据加载器上的 Top-1 和 Top-5 准确率。
    假设模型和数据已经在正确的设备上。
    """
    model.eval()  # 确保模型处于评估模式
    num_correct = 0
    num_samples = 0
    top5_num_correct = 0

    # 确定模型所在的设备，以便将数据移动到同一设备（如果尚未移动）
    # 通常，在调用此函数前，模型和数据加载器中的数据应已在同一设备
    # 但作为一种健壮性措施，我们可以检查并尝试移动数据
    # 不过，更推荐的做法是在外部处理好设备问题
    # model_device = next(model.parameters()).device # 获取模型参数所在的设备

    with torch.no_grad():  # 关闭梯度计算
        for x, y in loader:
            # 假设 x 和 y 已经通过 DataLoader 的 collate_fn 或外部逻辑放到了正确的设备
            # 如果需要在此处确保，可以这样做：
            # x = x.to(model_device, non_blocking=True)
            # y = y.to(model_device, non_blocking=True)
            
            scores = model(x)  # 前向传播
            # scores: [batch_size, num_classes]

            # Top-1 准确率
            _, predictions = scores.max(1)  # 获取 Top-1 预测的索引
            num_correct += (predictions == y).sum().item() # .item() 获取 Python 数字
            num_samples += predictions.size(0)

            # Top-5 准确率
            # scores.topk(k, dim) 返回 (values, indices)
            _, top5_indices = scores.topk(5, dim=1)  # 获取 Top-5 预测的索引
            
            # 将 y 扩展以匹配 top5_indices 的形状，方便比较
            # y: [batch_size] -> y_reshaped: [batch_size, 1]
            y_reshaped = y.view(-1, 1) 
            
            # 检查真实标签是否存在于 Top-5 预测中
            # (top5_indices == y_reshaped) 会进行逐元素比较，如果 y 中的值在 top5_indices 的对应行中，则为 True
            # .any(dim=1) 检查每行是否有至少一个 True
            top5_correct_batch = (top5_indices == y_reshaped).any(dim=1).sum().item()
            top5_num_correct += top5_correct_batch

    if num_samples == 0:
        print("警告: check_accuracy 接收到的样本数量为0。")
        return 0.0, 0.0

    acc = float(num_correct) / num_samples * 100
    top5_acc = float(top5_num_correct) / num_samples * 100

    # model.train() # 通常在评估函数后不立即切换回 train() 模式，调用者应根据需要处理
    return acc, top5_acc
