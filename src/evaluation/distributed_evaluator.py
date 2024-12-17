"""分布式评估器"""
import ray
from typing import Dict, List
import numpy as np

@ray.remote
class EvaluationWorker:
    """评估工作器"""
    def __init__(self, config: Dict):
        self.config = config
        self.env = self._create_env()
        
    def evaluate_episode(self, model_weights: Dict) -> Dict:
        """评估单个回合"""
        # 加载模型权重
        self.model.load_state_dict(model_weights)
        
        # 运行评估
        obs = self.env.reset()
        done = False
        total_reward = 0
        info = {}
        
        while not done:
            action = self.model.predict(obs)
            obs, reward, done, step_info = self.env.step(action)
            total_reward += reward
            info.update(step_info)
            
        return {
            'reward': total_reward,
            'info': info
        }

class DistributedEvaluator:
    """分布式评估器"""
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化Ray
        ray.init()
        
        # 创建工作器
        self.workers = [
            EvaluationWorker.remote(config)
            for _ in range(config['num_workers'])
        ]
        
    def evaluate(self, model_weights: Dict) -> Dict:
        """并行评估"""
        # 分发评估任务
        futures = [
            worker.evaluate_episode.remote(model_weights)
            for worker in self.workers
        ]
        
        # 收集结果
        results = ray.get(futures)
        
        # 汇总结果
        return self._aggregate_results(results) 