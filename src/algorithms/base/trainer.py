from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import time


class BaseTrainer(ABC):
    """训练器基类"""
    def __init__(self, env, agent, config: Dict):
        self.env = env
        self.agent = agent
        self.config = config
        
        # 训练配置
        self.total_steps = config.get('total_steps', 1000000)
        self.eval_interval = config.get('eval_interval', 10000)
        self.save_interval = config.get('save_interval', 50000)
        self.log_interval = config.get('log_interval', 1000)
        
        # 设置日志
        self.exp_name = config.get('exp_name', time.strftime('%Y%m%d_%H%M%S'))
        self.log_dir = os.path.join('experiments/logs', self.exp_name)
        self.writer = SummaryWriter(self.log_dir)
        
        # 保存路径
        self.save_dir = os.path.join('experiments/checkpoints', self.exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self):
        """训练主循环"""
        step = 0
        episode = 0
        
        while step < self.total_steps:
            # 重置环境
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # 选择动作
                action = self.agent.predict(state)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.store_transition(state, action, reward, next_state, done, info)
                
                episode_reward += reward
                episode_length += 1
                step += 1
                
                # 更新智能体
                if self.should_update(step):
                    metrics = self.update()
                    self.log_update(step, metrics)
                
                # 评估
                if step % self.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    self.log_eval(step, eval_metrics)
                
                # 保存模型
                if step % self.save_interval == 0:
                    self.save_checkpoint(step)
                
                if done:
                    # 记录回合信息
                    self.log_episode(episode, episode_reward, episode_length)
                    episode += 1
                    break
                
                state = next_state
                
        # 训练结束,保存最终模型
        self.save_checkpoint('final')
        
    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done, info):
        """存储经验"""
        pass
        
    @abstractmethod
    def should_update(self, step: int) -> bool:
        """是否应该更新"""
        pass
        
    @abstractmethod
    def update(self) -> Dict:
        """更新模型"""
        pass
        
    def evaluate(self, n_episodes=10) -> Dict:
        """评估智能体"""
        rewards = []
        lengths = []
        success_rate = 0
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.agent.predict(state)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    if info.get('success', False):
                        success_rate += 1
                    break
                    
                state = next_state
                
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
        return {
            'eval/mean_reward': np.mean(rewards),
            'eval/mean_length': np.mean(lengths),
            'eval/success_rate': success_rate / n_episodes
        }
        
    def save_checkpoint(self, step):
        """保存检查点"""
        path = os.path.join(self.save_dir, f'checkpoint_{step}.pt')
        self.agent.save(path)
        
    def log_update(self, step: int, metrics: Dict):
        """记录更新信息"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value, step)
            
    def log_eval(self, step: int, metrics: Dict):
        """记录评估信息"""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
            
    def log_episode(self, episode: int, reward: float, length: int):
        """记录回合信息"""
        self.writer.add_scalar('train/episode_reward', reward, episode)
        self.writer.add_scalar('train/episode_length', length, episode) 