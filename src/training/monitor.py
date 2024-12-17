import time
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import deque

class TrainingMonitor:
    """训练监控器"""
    def __init__(self, config: Dict):
        self.config = config
        
        # 性能指标
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
        # 训练指标
        self.value_losses = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        self.entropies = deque(maxlen=100)
        
        # 资源使用
        self.gpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        
        # 时间统计
        self.start_time = time.time()
        self.step_times = deque(maxlen=100)
        
    def update(self, stats: Dict):
        """更新统计信息"""
        # 更新性能指标
        if 'episode_reward' in stats:
            self.episode_rewards.append(stats['episode_reward'])
        if 'episode_length' in stats:
            self.episode_lengths.append(stats['episode_length'])
        if 'success' in stats:
            self.success_rate.append(float(stats['success']))
            
        # 更新训练指标
        if 'value_loss' in stats:
            self.value_losses.append(stats['value_loss'])
        if 'policy_loss' in stats:
            self.policy_losses.append(stats['policy_loss'])
        if 'entropy' in stats:
            self.entropies.append(stats['entropy'])
            
        # 更新资源使用
        if 'gpu_usage' in stats:
            self.gpu_usage.append(stats['gpu_usage'])
        if 'memory_usage' in stats:
            self.memory_usage.append(stats['memory_usage'])
            
        # 更新时间
        if 'step_time' in stats:
            self.step_times.append(stats['step_time'])
            
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        return {
            'avg_reward': np.mean(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'success_rate': np.mean(self.success_rate),
            'avg_value_loss': np.mean(self.value_losses),
            'avg_policy_loss': np.mean(self.policy_losses),
            'avg_entropy': np.mean(self.entropies),
            'avg_step_time': np.mean(self.step_times),
            'total_time': time.time() - self.start_time
        } 
    
    def analyze_performance(self) -> Dict:
        """分析训练性能"""
        analysis = {}
        
        # 计算收敛速度
        analysis['convergence_speed'] = self._analyze_convergence()
        
        # 计算稳定性
        analysis['stability'] = self._analyze_stability()
        
        # 计算资源效率
        analysis['resource_efficiency'] = self._analyze_resource_usage()
        
        return analysis
        
    def _analyze_convergence(self) -> Dict:
        """分析收敛性能"""
        # 计算奖励增长率
        rewards = np.array(self.episode_rewards)
        growth_rate = (rewards[-10:].mean() - rewards[:10].mean()) / len(rewards)
        
        # 检测收敛点
        window_size = 20
        means = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        convergence_step = np.argmin(np.abs(means[1:] - means[:-1])) + window_size
        
        return {
            'growth_rate': growth_rate,
            'convergence_step': convergence_step,
            'final_performance': rewards[-100:].mean()
        }
        
    def _analyze_stability(self) -> Dict:
        """分析训练稳定性"""
        # 计算奖励波动
        rewards = np.array(self.episode_rewards)
        reward_std = rewards[-100:].std()
        
        # 计算损失波动
        value_loss_std = np.array(self.value_losses).std()
        policy_loss_std = np.array(self.policy_losses).std()
        
        return {
            'reward_stability': 1.0 / (1.0 + reward_std),
            'value_loss_stability': 1.0 / (1.0 + value_loss_std),
            'policy_loss_stability': 1.0 / (1.0 + policy_loss_std)
        } 

    def _analyze_resource_usage(self) -> Dict:
        """分析资源使用效率"""
        # 计算GPU使用率
        gpu_usage = np.array(self.gpu_usage)
        avg_gpu = gpu_usage.mean()
        peak_gpu = gpu_usage.max()
        
        # 计算内存使用率
        memory_usage = np.array(self.memory_usage)
        avg_memory = memory_usage.mean()
        peak_memory = memory_usage.max()
        
        # 计算训练速度
        steps_per_second = len(self.step_times) / sum(self.step_times)
        
        # 计算资源效率分数
        gpu_efficiency = 1.0 - abs(0.8 - avg_gpu/100.0)  # 期望使用率80%
        memory_efficiency = 1.0 - avg_memory/peak_memory  # 内存使用稳定性
        
        return {
            'gpu_usage': {
                'average': avg_gpu,
                'peak': peak_gpu,
                'efficiency': gpu_efficiency
            },
            'memory_usage': {
                'average': avg_memory,
                'peak': peak_memory,
                'efficiency': memory_efficiency
            },
            'training_speed': {
                'steps_per_second': steps_per_second,
                'time_per_step': 1.0/steps_per_second
            }
        }

    def plot_metrics(self, save_path: str = None):
        """绘制训练指标"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制奖励曲线
        axes[0,0].plot(self.episode_rewards)
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].grid(True)
        
        # 绘制损失曲线
        axes[0,1].plot(self.value_losses, label='Value Loss')
        axes[0,1].plot(self.policy_losses, label='Policy Loss')
        axes[0,1].set_title('Training Losses')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 绘制成功率
        axes[1,0].plot(self.success_rate)
        axes[1,0].set_title('Success Rate')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Rate')
        axes[1,0].grid(True)
        
        # 绘制资源使用
        axes[1,1].plot(self.gpu_usage, label='GPU')
        axes[1,1].plot(self.memory_usage, label='Memory')
        axes[1,1].set_title('Resource Usage')
        axes[1,1].set_xlabel('Step')
        axes[1,1].set_ylabel('Usage (%)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def analyze_training_dynamics(self) -> Dict:
        """分析训练动态特性"""
        # 计算奖励变化趋势
        rewards = np.array(self.episode_rewards)
        reward_trend = np.polyfit(np.arange(len(rewards)), rewards, 1)[0]
        
        # 计算损失变化趋势
        value_losses = np.array(self.value_losses)
        policy_losses = np.array(self.policy_losses)
        
        value_trend = np.polyfit(np.arange(len(value_losses)), value_losses, 1)[0]
        policy_trend = np.polyfit(np.arange(len(policy_losses)), policy_losses, 1)[0]
        
        # 计算学习进展
        progress = {
            'early_stage': rewards[:len(rewards)//3].mean(),
            'mid_stage': rewards[len(rewards)//3:2*len(rewards)//3].mean(),
            'late_stage': rewards[2*len(rewards)//3:].mean()
        }
        
        # 计算训练效率
        efficiency = {
            'reward_per_step': rewards.sum() / len(self.step_times),
            'time_per_improvement': len(self.step_times) / (progress['late_stage'] - progress['early_stage'])
        }
        
        return {
            'trends': {
                'reward': reward_trend,
                'value_loss': value_trend,
                'policy_loss': policy_trend
            },
            'progress': progress,
            'efficiency': efficiency
        }

    def generate_training_report(self, save_path: str):
        """生成训练报告"""
        # 收集所有分析结果
        results = {
            'performance': self.analyze_performance(),
            'dynamics': self.analyze_training_dynamics(),
            'resources': self._analyze_resource_usage()
        }
        
        # 创建报告
        report = []
        report.append("# 训练报告\n")
        
        # 添加性能摘要
        report.append("## 性能摘要")
        report.append(f"- 最终奖励: {results['performance']['convergence_speed']['final_performance']:.2f}")
        report.append(f"- 收敛步数: {results['performance']['convergence_speed']['convergence_step']}")
        report.append(f"- 训练稳定性: {results['performance']['stability']['reward_stability']:.2f}\n")
        
        # 添加训练动态
        report.append("## 训练动态")
        report.append(f"- 奖励趋势: {results['dynamics']['trends']['reward']:.2e}")
        report.append(f"- 训练效率: {results['dynamics']['efficiency']['reward_per_step']:.2e}\n")
        
        # 添加资源使用
        report.append("## 资源使用")
        report.append(f"- GPU平均使用率: {results['resources']['gpu_usage']['average']:.1f}%")
        report.append(f"- 内存平均使用率: {results['resources']['memory_usage']['average']:.1f}%")
        report.append(f"- 训练速度: {results['resources']['training_speed']['steps_per_second']:.1f} steps/s\n")
        
        # 保存报告
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))