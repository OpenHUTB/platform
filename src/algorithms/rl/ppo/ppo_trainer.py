import numpy as np
from typing import Dict, List
import torch
from collections import deque


from src.algorithms.base.trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    """PPO训练器"""
    def __init__(self, env, agent, config: Dict):
        super().__init__(env, agent, config)
        
        # PPO特定配置
        self.n_steps = config.get('n_steps', 2048)
        self.batch_size = config.get('batch_size', 64)
        
        # 经验缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
    def store_transition(self, state, action, reward, next_state, done, info):
        """存储经验"""
        # 获取当前状态的价值估计
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.agent.device)
            _, value = self.agent.network(state_tensor)
            
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value.cpu().numpy())
        self.buffer['dones'].append(done)
        
    def should_update(self, step: int) -> bool:
        """是否应该更新"""
        return len(self.buffer['states']) >= self.n_steps
        
    def update(self) -> Dict:
        """更新模型"""
        # 计算优势和回报
        advantages, returns = self._compute_gae()
        
        # 准备训练数据
        train_data = {
            'states': np.array(self.buffer['states']),
            'actions': np.array(self.buffer['actions']),
            'returns': returns,
            'advantages': advantages,
            'values': np.array(self.buffer['values']),
            'log_probs': np.array(self.buffer['log_probs'])
        }
        
        # 清空缓冲区
        self.buffer = {key: [] for key in self.buffer.keys()}
        
        # 更新模型
        metrics = self.agent.update(train_data)
        
        return metrics
        
    def _compute_gae(self):
        """计算广义优势估计(GAE)"""
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        dones = np.array(self.buffer['dones'])
        
        # 获取最后一个状态的价值估计
        with torch.no_grad():
            last_state = self.buffer['states'][-1]
            last_state_tensor = torch.FloatTensor(last_state).to(self.agent.device)
            _, last_value = self.agent.network(last_state_tensor)
            last_value = last_value.cpu().numpy()
            
        # 计算GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.agent.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.agent.gamma * self.agent.gae_lambda * (1 - dones[t]) * last_gae
            
        # 计算回报
        returns = advantages + values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns 