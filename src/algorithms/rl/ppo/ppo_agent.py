import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
import numpy as np

from src.algorithms.base import BaseAlgorithm


class PPONetwork(nn.Module):
    """PPO网络"""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()  # 输出范围[-1,1]
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(obs)
        action = self.actor(features)
        value = self.critic(features)
        return action, value


class PPOAgent(BaseAlgorithm):
    """PPO算法实现"""
    def __init__(self, config: Dict):
        super().__init__(config)
        
        self.obs_dim = config['obs_dim']
        self.act_dim = config['act_dim']
        self.hidden_dim = config.get('hidden_dim', 256)
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.epochs = config.get('epochs', 10)
        
        # 创建网络
        self.network = PPONetwork(
            self.obs_dim,
            self.act_dim,
            self.hidden_dim
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        """预测动作"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, _ = self.network(state)
        return action.cpu().numpy()
        
    def update(self, batch: Dict) -> Dict:
        """更新模型"""
        # 准备数据
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)
        old_values = torch.FloatTensor(batch['values']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        
        # 多轮训练
        for _ in range(self.epochs):
            # 获取当前策略的动作和价值
            new_actions, new_values = self.network(states)
            new_log_probs = self._compute_log_probs(new_actions, actions)
            
            # 计算策略比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算PPO目标
            obj1 = ratio * advantages
            obj2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * advantages
            policy_loss = -torch.min(obj1, obj2).mean()
            
            # 计算价值损失
            value_loss = nn.MSELoss()(new_values, returns)
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss
            
            # 更新网络
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item()
        }
        
    def _compute_log_probs(self, pred_actions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率"""
        # 简单起见,这里假设动作是确定性的
        return -((pred_actions - actions) ** 2).sum(-1)
        
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 