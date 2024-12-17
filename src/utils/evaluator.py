import numpy as np
from typing import Dict, List
import json
import os
from datetime import datetime


class ModelEvaluator:
    """模型评估器"""
    def __init__(self, env, agent, config: Dict):
        self.env = env
        self.agent = agent
        self.config = config
        
        # 评估配置
        self.n_episodes = config.get('n_episodes', 100)
        self.save_video = config.get('save_video', True)
        self.metrics_file = config.get('metrics_file', 'evaluation_metrics.json')
        
    def evaluate(self) -> Dict:
        """评估模型"""
        metrics = {
            'rewards': [],
            'lengths': [],
            'success_rate': 0,
            'collision_rate': 0,
            'timeout_rate': 0,
            'completion_rate': 0,
            'avg_speed': [],
            'avg_acceleration': [],
            'avg_jerk': [],
            'min_distance_to_others': []
        }
        
        for episode in range(self.n_episodes):
            episode_metrics = self._evaluate_episode()
            
            # 更新指标
            for key in metrics.keys():
                if isinstance(episode_metrics[key], (int, float)):
                    metrics[key].append(episode_metrics[key])
                elif isinstance(episode_metrics[key], bool):
                    metrics[key] += int(episode_metrics[key])
                    
        # 计算统计信息
        results = self._compute_statistics(metrics)
        
        # 保存结果
        self._save_results(results)
        
        return results
        
    def _evaluate_episode(self) -> Dict:
        """评估单个回合"""
        state = self.env.reset()
        done = False
        
        episode_metrics = {
            'reward': 0,
            'length': 0,
            'success': False,
            'collision': False,
            'timeout': False,
            'completion': 0,
            'speed': [],
            'acceleration': [],
            'jerk': [],
            'distance_to_others': []
        }
        
        while not done:
            action = self.agent.predict(state)
            next_state, reward, done, info = self.env.step(action)
            
            # 更新指标
            episode_metrics['reward'] += reward
            episode_metrics['length'] += 1
            episode_metrics['speed'].append(info.get('speed', 0))
            episode_metrics['acceleration'].append(info.get('acceleration', 0))
            episode_metrics['jerk'].append(info.get('jerk', 0))
            episode_metrics['distance_to_others'].append(info.get('distance_to_others', float('inf')))
            
            if done:
                episode_metrics['success'] = info.get('success', False)
                episode_metrics['collision'] = info.get('collision', False)
                episode_metrics['timeout'] = info.get('timeout', False)
                episode_metrics['completion'] = info.get('completion', 0)
                
            state = next_state
            
        return episode_metrics
        
    def _compute_statistics(self, metrics: Dict) -> Dict:
        """计算统计信息"""
        results = {}
        
        # 计算平均值和标准差
        for key, values in metrics.items():
            if isinstance(values, list) and values:
                results[f'{key}/mean'] = np.mean(values)
                results[f'{key}/std'] = np.std(values)
                results[f'{key}/min'] = np.min(values)
                results[f'{key}/max'] = np.max(values)
            else:
                results[key] = values / self.n_episodes if isinstance(values, (int, float)) else values
                
        return results
        
    def _save_results(self, results: Dict):
        """保存评估结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join('experiments/results', f'eval_{timestamp}.json')
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4) 