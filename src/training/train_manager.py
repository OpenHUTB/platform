import os
import time
from typing import Dict, Any
import torch
from torch.utils.tensorboard import SummaryWriter

from src.environments.rl_env import RLEnv
from src.algorithms.registry import ALGORITHM_REGISTRY
from src.utils.logger import Logger
from src.utils.evaluator import ModelEvaluator
from src.visualization.dashboard.dashboard_app import DashboardApp


class TrainManager:
    """训练管理器"""
    def __init__(self, config: Dict):
        self.config = config
        
        # 创建环境
        self.env = RLEnv(config['environment'])
        self.eval_env = RLEnv(config['environment'])
        
        # 创建算法
        algo_name = config['algorithm']['name']
        algo_class = ALGORITHM_REGISTRY.get(algo_name)
        self.agent = algo_class(config['algorithm'])
        
        # 创建评估器
        self.evaluator = ModelEvaluator(self.eval_env, self.agent, config['evaluation'])
        
        # 创建日志记录器
        self.logger = Logger(config['logging'])
        
        # 创建可视化面板
        if config['visualization']['enabled']:
            self.dashboard = DashboardApp(config['visualization'])
        
        # 训练状态
        self.total_steps = 0
        self.episodes = 0
        self.best_reward = float('-inf')
        
    def train(self):
        """训练主循环"""
        start_time = time.time()
        
        while self.total_steps < self.config['training']['max_steps']:
            # 运行一个回合
            episode_stats = self._run_episode()
            
            # 记录日志
            self.logger.log_episode(episode_stats)
            
            # 更新可视化
            if hasattr(self, 'dashboard'):
                self.dashboard.update(episode_stats)
            
            # 定期评估
            if self.total_steps % self.config['evaluation']['interval'] == 0:
                eval_stats = self.evaluator.evaluate()
                self.logger.log_evaluation(eval_stats)
                
                # 保存最佳模型
                if eval_stats['mean_reward'] > self.best_reward:
                    self.best_reward = eval_stats['mean_reward']
                    self._save_checkpoint('best')
            
            # 定期保存检查点
            if self.total_steps % self.config['training']['save_interval'] == 0:
                self._save_checkpoint(f'step_{self.total_steps}')
            
            self.episodes += 1
            
        # 训练结束
        total_time = time.time() - start_time
        self.logger.log_training_end({
            'total_steps': self.total_steps,
            'total_episodes': self.episodes,
            'total_time': total_time,
            'best_reward': self.best_reward
        })
        
    def _run_episode(self) -> Dict:
        """运行一个训练回合"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_info = {}
        
        while True:
            # 选择动作
            action = self.agent.predict(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.agent.store_transition(state, action, reward, next_state, done, info)
            
            # 更新智能体
            if self.agent.should_update(self.total_steps):
                update_info = self.agent.update()
                episode_info.update(update_info)
            
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            if done:
                break
                
            state = next_state
            
        return {
            'reward': episode_reward,
            'length': episode_length,
            'total_steps': self.total_steps,
            **episode_info
        }
        
    def _save_checkpoint(self, tag: str):
        """保存检查点"""
        save_path = os.path.join(self.config['training']['checkpoint_dir'], f'checkpoint_{tag}.pt')
        self.agent.save(save_path)
        
        # 保存训练状态
        state_path = os.path.join(self.config['training']['checkpoint_dir'], f'training_state_{tag}.pt')
        torch.save({
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'best_reward': self.best_reward
        }, state_path) 