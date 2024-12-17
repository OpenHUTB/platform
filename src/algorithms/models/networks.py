import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            ])
            prev_dim = hidden_size
            
        # 输出层
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, output_dim)
        self.log_std_head = nn.Linear(prev_dim, output_dim)
        
        # 初始化
        self.apply(self._init_weights)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.backbone(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
        
    def _init_weights(self, m: nn.Module):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
            
class ValueNetwork(nn.Module):
    """值函数网络"""
    def __init__(self, input_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            ])
            prev_dim = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # 初始化
        self.apply(self._init_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(x)
        
    def _init_weights(self, m: nn.Module):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0) 