import torch
import torch.nn as nn
from typing import Dict, Any

from src.algorithms.base import BaseAlgorithm, BaseNetwork
from src.utils.registry import register_algorithm

class CustomEncoder(BaseNetwork):
    """自定义编码器"""
    def __init__(self, config: Dict):
        super().__init__()
        # 定义网络结构
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.mlp(x)
        return x
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

@register_algorithm("custom")
class CustomAlgorithm(BaseAlgorithm):
    """自定义算法"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 创建网络
        self.encoder = CustomEncoder(config['encoder'])
        self.policy = self._create_policy(config['policy'])
        self.value = self._create_value(config['value'])
        
        # 创建优化器
        self.optimizer = self._create_optimizer(config['optimizer'])
        
        # 其他设置
        self.device = torch.device(config.get('device', 'cuda'))
        self.to(self.device)
        
    def predict(self, observation: Dict[str, torch.Tensor]) -> np.ndarray:
        """预测动作"""
        with torch.no_grad():
            # 编码观察
            obs = self.process_observation(observation)
            features = self.encoder(obs)
            
            # 获取动作分布
            action_dist = self.policy(features)
            
            # 采样动作
            if self.training:
                action = action_dist.sample()
            else:
                action = action_dist.mean
                
        return action.cpu().numpy()
        
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """更新算法"""
        # 处理数据
        processed_batch = self.process_experience(batch)
        
        # 计算损失
        policy_loss = self._compute_policy_loss(processed_batch)
        value_loss = self._compute_value_loss(processed_batch)
        
        # 更新网络
        self.optimizer.zero_grad()
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        } 