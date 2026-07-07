# 定义一个字典，存储不同优化器的默认超参数配置
DEFAULT_OPT_PARAMS = {
    # SGD 优化器的默认参数
    "sgd": {
        "first_momentum": 0.0,          # 一阶动量（Momentum），通常用于 SGD with Momentum
        "second_momentum": 0.0,         # 二阶动量，SGD 通常不使用，可能为扩展或兼容性保留
        "dampening": 0.0,               # 动量的抑制因子
        "weight_decay": 0.0,            # 权重衰减（L2 正则化）系数
        "lmbda": 1e-3,                  # 可能与特定正则化或重要性加权相关的 lambda 参数
        "lmbda_amplify": 2,             # lambda 参数的放大系数，可能用于调整其影响
        "hat_lmbda_coeff": 10           # 可能与 lambda 的某种估计或调整相关的系数
    }
    ,
    # Adam 优化器的默认参数
    "adam": {
        "lr": 1e-3,                     # 学习率
        "first_momentum": 0.9,          # 一阶矩估计的指数衰减率 (beta1)
        "second_momentum": 0.999,       # 二阶矩估计的指数衰减率 (beta2)
        "dampening": 0.0,               # 抑制因子，Adam 通常不使用
        "weight_decay": 0.0,            # 权重衰减系数，标准 Adam 通常在优化步骤之外应用
        "lmbda": 1e-2,                  # 可能与特定正则化或重要性加权相关的 lambda 参数
        "lmbda_amplify": 20,            # lambda 参数的放大系数
        "hat_lmbda_coeff": 1e3          # 可能与 lambda 的某种估计或调整相关的系数
    }
    ,
    # AdamW 优化器的默认参数 (Adam with decoupled weight decay)
    "adamw": {
        "lr": 1e-3,                     # 学习率
        "first_momentum": 0.9,          # 一阶矩估计的指数衰减率 (beta1)
        "second_momentum": 0.999,       # 二阶矩估计的指数衰减率 (beta2)
        "dampening": 0.0,               # 抑制因子，AdamW 通常不使用
        "weight_decay": 1e-2,           # 权重衰减系数，AdamW 将其与梯度更新解耦
        "lmbda": 1e-2,                  # 可能与特定正则化或重要性加权相关的 lambda 参数
        "lmbda_amplify": 20,            # lambda 参数的放大系数
        "hat_lmbda_coeff": 1e3          # 可能与 lambda 的某种估计或调整相关的系数
    }
}

# 定义一个列表，包含支持的梯度估计方法（或优化器类型）的名称
# 这些名称对应于 DEFAULT_OPT_PARAMS 字典中的键
SUPPORT_GRADIENT_ESTIMATES = ['sgd', 'adam', 'adamw']
