"""基准测试套件"""
from typing import Dict, List
import yaml
import json
from src.environments.tasks.task_registry import TASK_REGISTRY

class BenchmarkSuite:
    """基准测试套件"""
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 创建任务环境
        self.tasks = {}
        for task_name, task_config in self.config['tasks'].items():
            task_class = TASK_REGISTRY.get(task_config['type'])
            self.tasks[task_name] = task_class(task_config)
            
    def evaluate(self, agent, save_path: str = None) -> Dict:
        """评估智能体"""
        results = {}
        
        # 在每个任务上评估
        for task_name, env in self.tasks.items():
            task_results = self._evaluate_task(env, agent)
            results[task_name] = task_results
            
        # 计算总体得分
        results['overall'] = self._compute_overall_score(results)
        
        # 保存结果
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
                
        return results
        
    def _evaluate_task(self, env, agent) -> Dict:
        """评估单个任务"""
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        
        for _ in range(self.config['n_episodes']):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action = agent.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            if info.get('success', False):
                success_rate += 1
                
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'success_rate': float(success_rate / self.config['n_episodes'])
        }
        
    def _compute_overall_score(self, results: Dict) -> float:
        """计算总体得分"""
        # 根据配置的权重计算加权分数
        weights = self.config.get('task_weights', {})
        total_score = 0
        
        for task_name, task_results in results.items():
            if task_name == 'overall':
                continue
            weight = weights.get(task_name, 1.0)
            score = task_results['mean_reward'] * weight
            total_score += score
            
        return total_score 