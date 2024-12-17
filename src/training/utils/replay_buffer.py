import numpy as np
from typing import Dict, List, Tuple
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, state: Dict, action: np.ndarray, reward: float, 
            next_state: Dict, done: bool, info: Dict):
        """添加经验"""
        experience = (state, action, reward, next_state, done, info)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(1.0)  # 新经验的优先级最高
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = 1.0
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int = None) -> Tuple:
        """采样经验"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # 根据优先级采样
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 获取经验
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        infos = []
        
        for idx in indices:
            state, action, reward, next_state, done, info = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            infos.append(info)
            
        return {
            'states': states,
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': next_states,
            'dones': np.array(dones),
            'infos': infos,
            'indices': indices
        }
        
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # 避免优先级为0
            
    def __len__(self) -> int:
        return len(self.buffer) 