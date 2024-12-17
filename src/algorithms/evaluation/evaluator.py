"""算法评估工具"""
from typing import Dict, List, Optional
import numpy as np
import json
import os
from datetime import datetime
from src.utils.logger import MetricLogger

class AlgorithmEvaluator:
    """算法评估器"""
    def __init__(self, env, agent, config: Dict):
        self.env = env
        self.agent = agent
        self.config = config
        self.logger = MetricLogger()
        
        # 评估配置
        self.n_episodes = config.get('n_episodes', 10)
        self.save_video = config.get('save_video', True)
        self.metrics = config.get('metrics', ['reward', 'length', 'success_rate'])
        
    def evaluate(self, save_path: Optional[str] = None) -> Dict:
        """评估算法"""
        all_metrics = []
        
        for episode in range(self.n_episodes):
            # 运行一个回合
            episode_metrics = self._run_episode()
            all_metrics.append(episode_metrics)
            
            # 记录日志
            self.logger.log_metrics(episode_metrics, episode)
            
        # 计算统计结果
        results = self._compute_statistics(all_metrics)
        
        # 保存结果
        if save_path:
            self._save_results(results, save_path)
            
        return results
        
    def _run_episode(self) -> Dict:
        """运行单个回合"""
        state = self.env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        
        metrics = {
            'reward': 0,
            'length': 0,
            'success': False,
            'collision': False,
            'timeout': False,
            'distance_traveled': 0,
            'average_speed': 0,
            'min_distance_to_obstacles': float('inf')
        }
        
        while not done:
            # 选择动作
            action = self.agent.predict(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 更新指标
            total_reward += reward
            episode_length += 1
            metrics = self._update_metrics(metrics, state, action, reward, next_state, info)
            
            state = next_state
            
        # 完成回合
        metrics['reward'] = total_reward
        metrics['length'] = episode_length
        metrics['success'] = info.get('success', False)
        
        return metrics
        
    def _update_metrics(self, metrics: Dict, state: np.ndarray, action: np.ndarray,
                       reward: float, next_state: np.ndarray, info: Dict) -> Dict:
        """更新评估指标"""
        # 更新距离
        if 'distance_traveled' in info:
            metrics['distance_traveled'] += info['distance_traveled']
            
        # 更新速度
        if 'speed' in info:
            metrics['average_speed'] = (metrics['average_speed'] * (metrics['length']) + 
                                      info['speed']) / (metrics['length'] + 1)
            
        # 更新障碍物距离
        if 'distance_to_obstacles' in info:
            metrics['min_distance_to_obstacles'] = min(
                metrics['min_distance_to_obstacles'],
                info['distance_to_obstacles']
            )
            
        return metrics
        
    def _compute_statistics(self, metrics_list: List[Dict]) -> Dict:
        """计算统计结果"""
        results = {}
        
        # 计算每个指标的统计值
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            results[f'{key}/mean'] = np.mean(values)
            results[f'{key}/std'] = np.std(values)
            results[f'{key}/min'] = np.min(values)
            results[f'{key}/max'] = np.max(values)
            
        # 计算成功率
        results['success_rate'] = np.mean([m['success'] for m in metrics_list])
        
        return results
        
    def _save_results(self, results: Dict, save_path: str):
        """保存评估结果"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 添加元信息
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': self.agent.__class__.__name__,
            'environment': self.env.__class__.__name__,
            'n_episodes': self.n_episodes,
            'config': self.config
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4) 