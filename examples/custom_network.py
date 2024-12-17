import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.algorithms.base import BaseNetwork
from src.utils.registry import register_network


@register_network("custom_encoder")
class CustomEncoder(BaseNetwork):
    """自定义编码器网络"""
    def __init__(self, config: Dict):
        super().__init__()
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        
        # 激光雷达编码器
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU()
        )
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 编码图像
        img_features = self.image_encoder(obs['rgb'])
        
        # 编码激光雷达
        lidar_features = self.lidar_encoder(obs['lidar'])
        
        # 编码状态
        state_features = self.state_encoder(obs['state'])
        
        # 特征融合
        features = torch.cat([img_features, lidar_features, state_features], dim=1)
        features = self.fusion(features)
        
        return features
        
    def reset_parameters(self):
        """重置参数"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias) 