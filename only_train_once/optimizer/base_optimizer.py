from abc import abstractclassmethod  # 导入抽象类方法装饰器，用于定义必须在子类中实现的方法
from torch.optim.optimizer import Optimizer, required  # 导入PyTorch优化器基类和required标记
import torch  # 导入PyTorch库

class BaseOptimizer(Optimizer):
    """
    优化器的基类，继承自PyTorch的Optimizer类。
    实现了常见优化算法的基本组件，如动量计算和梯度变体的生成。
    """
    def __init__(self, params, defaults=dict(), **kwargs):
        """
        初始化优化器
        
        参数:
            params: 待优化参数的迭代器
            defaults: 优化器默认参数字典
            **kwargs: 额外的关键字参数
        """
        super(BaseOptimizer, self).__init__(params, defaults)  # 调用父类初始化方法
        self.num_steps = 0  # 优化步数计数器，用于偏差校正
        self.safe_guard = 1e-8  # 安全系数，防止除零错误
        self.first_moment_grads = dict()  # 存储一阶动量的字典
        self.second_moment_grads = dict()  # 存储二阶动量的字典
        
    def get_second_momentum_grad_square(self, name, second_moment, dampening, grad):
        """
        计算并返回梯度的二阶动量(平方)
        
        参数:
            name: 动量缓存的唯一标识符
            second_moment: 二阶动量系数(beta2)
            dampening: 阻尼系数
            grad: 当前梯度
            
        返回:
            更新后的二阶动量
        """
        if second_moment > 0:  # 如果需要使用二阶动量
            if name not in self.second_moment_grads:  # 如果是首次计算该参数的动量
                buf = self.second_moment_grads[name] = grad * grad  # 初始化动量为梯度的平方
            else:  # 已经有历史动量
                buf = self.second_moment_grads[name]  # 获取历史动量
                buf.mul_(second_moment).add_(grad * grad, alpha=(1.0-dampening))  # 更新动量: second_moment * old + (1-dampening) * grad^2
            return buf
        else:  # 如果不使用动量
            return grad * grad  # 直接返回梯度的平方

    def __setstate__(self, state):
        """
        设置优化器状态，用于加载保存的优化器
        
        参数:
            state: 优化器状态字典
        """
        super(BaseOptimizer, self).__setstate__(state)  # 调用父类的状态恢复方法

    def get_first_momentum_grad(self, name, first_moment, dampening, grad):
        """
        计算并返回梯度的一阶动量
        
        参数:
            name: 动量缓存的唯一标识符
            first_moment: 一阶动量系数(beta1)
            dampening: 阻尼系数
            grad: 当前梯度
            
        返回:
            更新后的一阶动量
        """
        if first_moment > 0:  # 如果需要使用一阶动量
            if name not in self.first_moment_grads:  # 如果是首次计算该参数的动量
                buf = self.first_moment_grads[name] = grad  # 初始化动量为梯度
            else:  # 已经有历史动量
                buf = self.first_moment_grads[name]  # 获取历史动量
                buf.mul_(first_moment).add_(grad, alpha=(1.0-dampening))  # 更新动量: first_moment * old + (1-dampening) * grad
            return buf
        else:  # 如果不使用动量
            return grad  # 直接返回原始梯度

    def compute_grad_variant(self):
        """
        计算各种优化算法的梯度变体。
        支持Adam/AdamW和基于动量的SGD等优化算法。
        """
        for i, group in enumerate(self.param_groups):  # 遍历所有参数组
            # 确定是否使用Adam/AdamW优化器
            is_adam = group['variant'] == 'adam' or group['variant'] == 'adamw'
            # 计算Adam的偏差校正因子
            first_bias_correction = 1.0 - group['first_momentum'] ** self.num_steps if is_adam else None
            second_bias_correction = 1.0 - group['second_momentum'] ** self.num_steps if is_adam else None
            
            group['grad_variant'] = dict()  # 初始化梯度变体字典
            
            # 遍历参数组中的每个参数
            for j, (p_name, p) in enumerate(zip(group['p_names'], group['params'])):
                if p.grad is None:  # 跳过没有梯度的参数
                    continue
                
                # 克隆梯度并分离，确保不会影响原始梯度
                refined_grad_f = torch.clone(p.grad.data).detach()
                
                # 应用权重衰减(除了AdamW，它在参数更新时单独处理权重衰减)
                if group['weight_decay'] is not None and group['variant'] != 'adamw':
                    refined_grad_f += group['weight_decay'] * p.data
                
                if not is_adam:  # SGD或其他非Adam类优化器
                    # 应用一阶动量(如果需要)
                    if group['first_momentum'] > 0.0 or group['dampening'] > 0.0:
                        refined_grad_f = self.get_first_momentum_grad(f"grad_first_moment_buffer_group_{i}_param_{j}", 
                            group['first_momentum'], group['dampening'], refined_grad_f)
                    group['grad_variant'][p_name] = refined_grad_f
                else:  # Adam或AdamW优化器
                    # 计算一阶动量(m_t)
                    first_moment_grad = self.get_first_momentum_grad(f"grad_first_moment_buffer_group_{i}_param_{j}", 
                        group['first_momentum'], group['first_momentum'], refined_grad_f) 
                    # 计算二阶动量(v_t)
                    second_moment_grad_sq = self.get_second_momentum_grad_square(f"grad_second_moment_buffer_group_{i}_param_{j}", 
                        group['second_momentum'], group['second_momentum'], refined_grad_f)

                    # 应用偏差校正
                    exp_avg_first_moment_grad = first_moment_grad / first_bias_correction
                    exp_avg_second_moment_grad_sq = second_moment_grad_sq / second_bias_correction
                    
                    # 计算Adam更新方向: m_t / (sqrt(v_t) + epsilon)
                    denom = exp_avg_second_moment_grad_sq.sqrt().add_(self.safe_guard)
                    group['grad_variant'][p_name] = exp_avg_first_moment_grad / denom

    def set_learning_rate(self, lr):
        """
        设置所有参数组的学习率
        
        参数:
            lr: 新的学习率
        """
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        """
        获取当前学习率(返回第一个参数组的学习率)
        
        返回:
            当前学习率
        """
        for param_group in self.param_groups:
            lr = param_group['lr']
        return lr
    
    @abstractclassmethod
    def step(self, loss=None):
        """
        执行单步优化。这是一个抽象方法，需要在子类中实现。
        
        参数:
            loss: 可选的损失值
            
        注意:
            子类必须实现此方法来执行参数更新
        """
        raise NotImplementedError  # 子类必须实现此方法