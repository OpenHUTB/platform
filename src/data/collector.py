"""数据收集器"""
from typing import Dict, List
import numpy as np
import h5py
import json
from pathlib import Path

class DataCollector:
    """数据收集器"""
    def __init__(self, config: Dict):
        self.config = config
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据缓存
        self.episode_buffer = []
        self.current_episode = 0
        
    def collect(self, obs: Dict, action: np.ndarray, reward: float, 
                next_obs: Dict, done: bool, info: Dict):
        """收集数据"""
        data = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'info': info
        }
        self.episode_buffer.append(data)
        
        if done:
            self._save_episode()
            self.episode_buffer = []
            self.current_episode += 1
            
    def _save_episode(self):
        """保存回合数据"""
        episode_path = self.save_dir / f"episode_{self.current_episode:05d}.h5"
        
        with h5py.File(episode_path, 'w') as f:
            # 保存传感器数据
            for i, data in enumerate(self.episode_buffer):
                step_group = f.create_group(f"step_{i:05d}")
                
                # 保存观测
                for key, value in data['obs'].items():
                    step_group.create_dataset(f"obs/{key}", data=value)
                    
                # 保存动作和奖励
                step_group.create_dataset("action", data=data['action'])
                step_group.create_dataset("reward", data=data['reward'])
                
                # 保存下一个观测
                for key, value in data['next_obs'].items():
                    step_group.create_dataset(f"next_obs/{key}", data=value)
                    
                # 保存其他信息
                step_group.create_dataset("done", data=data['done'])
                step_group.attrs['info'] = json.dumps(data['info']) 