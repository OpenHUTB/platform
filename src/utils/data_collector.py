import numpy as np
from typing import Dict, List, Any
import h5py
import json
import os
from datetime import datetime
import threading
import queue
import time

class DataCollector:
    """数据收集器"""
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = config.get('data_dir', 'experiments/data')
        self.buffer_size = config.get('buffer_size', 1000)
        self.save_interval = config.get('save_interval', 100)
        
        # 数据缓冲区
        self.buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'infos': []
        }
        
        # 传感器数据缓冲区
        self.sensor_buffer = {
            'camera_rgb': [],
            'lidar': [],
            'gnss': [],
            'imu': []
        }
        
        # 创建保存目录
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 异步保存队列
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._async_save_worker)
        self.save_thread.daemon = True
        self.save_thread.start()
        
    def add_transition(self, obs: Dict, action: np.ndarray, reward: float,
                      next_obs: Dict, done: bool, info: Dict):
        """添加转换数据"""
        self.buffer['observations'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_observations'].append(next_obs)
        self.buffer['dones'].append(done)
        self.buffer['infos'].append(info)
        
        # 处理传感器数据
        for sensor_name in self.sensor_buffer.keys():
            if sensor_name in obs:
                self.sensor_buffer[sensor_name].append(obs[sensor_name])
                
        # 检查是否需要保存
        if len(self.buffer['observations']) >= self.buffer_size:
            self.save_queue.put(self._get_save_data())
            self._clear_buffer()
            
    def _get_save_data(self) -> Dict:
        """获取要保存的数据"""
        save_data = {
            'transitions': {
                'observations': np.array(self.buffer['observations']),
                'actions': np.array(self.buffer['actions']),
                'rewards': np.array(self.buffer['rewards']),
                'next_observations': np.array(self.buffer['next_observations']),
                'dones': np.array(self.buffer['dones'])
            },
            'sensor_data': {
                name: np.array(data) 
                for name, data in self.sensor_buffer.items()
                if data
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'buffer_size': len(self.buffer['observations']),
                'config': self.config
            }
        }
        return save_data
        
    def _clear_buffer(self):
        """清空缓冲区"""
        for key in self.buffer.keys():
            self.buffer[key] = []
        for key in self.sensor_buffer.keys():
            self.sensor_buffer[key] = []
            
    def _async_save_worker(self):
        """异步保存工作线程"""
        while True:
            try:
                save_data = self.save_queue.get()
                self._save_data(save_data)
                self.save_queue.task_done()
            except Exception as e:
                print(f"Error saving data: {e}")
            time.sleep(0.1)
            
    def _save_data(self, save_data: Dict):
        """保存数据"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存转换数据
        h5_path = os.path.join(self.data_dir, f'transitions_{timestamp}.h5')
        with h5py.File(h5_path, 'w') as f:
            # 保存转换数据
            transitions = f.create_group('transitions')
            for key, value in save_data['transitions'].items():
                transitions.create_dataset(key, data=value)
                
            # 保存传感器数据
            sensors = f.create_group('sensors')
            for name, data in save_data['sensor_data'].items():
                sensors.create_dataset(name, data=data)
                
        # 保存元数据
        meta_path = os.path.join(self.data_dir, f'metadata_{timestamp}.json')
        with open(meta_path, 'w') as f:
            json.dump(save_data['metadata'], f, indent=4)
            
    def close(self):
        """关闭数据收集器"""
        # 保存剩余数据
        if len(self.buffer['observations']) > 0:
            self.save_queue.put(self._get_save_data())
            
        # 等待所有数据保存完成
        self.save_queue.join() 