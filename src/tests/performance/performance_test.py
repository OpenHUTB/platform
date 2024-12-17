from typing import Dict, List
import carla
import numpy as np
import psutil
import GPUtil
from src.tests.base.test_base import BaseTest

class PerformanceTest(BaseTest):
    """性能测试"""
    def __init__(self, client: carla.Client, config: Dict):
        super().__init__(client, config)
        self.sampling_interval = config.get('sampling_interval', 1.0)  # 采样间隔(秒)
        self.metrics_history = []
        
    def setup(self):
        """准备测试"""
        # 初始化性能监控
        self.metrics_history = []
        
    def run(self):
        """运行性能测试"""
        test_duration = self.config.get('duration', 60)  # 测试时长(秒)
        elapsed_time = 0
        
        while elapsed_time < test_duration:
            # 收集性能指标
            metrics = self._collect_performance_metrics()
            self.metrics_history.append(metrics)
            
            # 等待下一个采样点
            time.sleep(self.sampling_interval)
            elapsed_time += self.sampling_interval
            
        # 计算统计指标
        self._compute_statistics()
        
    def cleanup(self):
        """清理测试"""
        pass
        
    def _collect_performance_metrics(self) -> Dict:
        """收集性能指标"""
        metrics = {}
        
        # CPU使用率
        metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
        
        # 内存使用
        memory = psutil.virtual_memory()
        metrics['memory_used'] = memory.used / (1024 ** 3)  # GB
        metrics['memory_percent'] = memory.percent
        
        # GPU使用
        gpus = GPUtil.getGPUs()
        if gpus:
            metrics['gpu_utilization'] = gpus[0].load * 100
            metrics['gpu_memory_used'] = gpus[0].memoryUsed
            metrics['gpu_memory_percent'] = gpus[0].memoryUtil * 100
            
        # 进程信息
        process = psutil.Process()
        metrics['process_cpu_percent'] = process.cpu_percent()
        metrics['process_memory_mb'] = process.memory_info().rss / (1024 * 1024)
        
        return metrics
        
    def _compute_statistics(self):
        """计算统计指标"""
        if not self.metrics_history:
            return
            
        # 转换为numpy数组便于计算
        metrics_array = {
            key: np.array([m[key] for m in self.metrics_history])
            for key in self.metrics_history[0].keys()
        }
        
        # 计算统计指标
        for key, values in metrics_array.items():
            self.results['metrics'][f'{key}_mean'] = np.mean(values)
            self.results['metrics'][f'{key}_std'] = np.std(values)
            self.results['metrics'][f'{key}_max'] = np.max(values)
            self.results['metrics'][f'{key}_min'] = np.min(values) 