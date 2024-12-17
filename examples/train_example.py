"""训练示例"""
import os
from src.environments import CarlaEnv
from src.algorithms.rl.ppo import PPOAgent, PPOTrainer
from src.utils.logger import TensorboardLogger

def main():
    # 环境配置
    env_config = {
        'carla': {
            'host': 'localhost',
            'port': 2000
        },
        'scenario': 'urban',  # 使用城市场景
        'sensors': {
            'camera_rgb': True,
            'lidar': True
        }
    }
    
    # 训练配置
    train_config = {
        'total_steps': 1000000,
        'eval_interval': 10000,
        'save_interval': 50000,
        'log_interval': 1000,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10
    }
    
    # 创建环境
    env = CarlaEnv(env_config)
    
    # 创建智能体
    agent = PPOAgent({
        'obs_dim': env.observation_space.shape[0],
        'act_dim': env.action_space.shape[0],
        'hidden_dim': 256,
        'learning_rate': 3e-4
    })
    
    # 创建日志记录器
    logger = TensorboardLogger(log_dir='experiments/logs/ppo_urban')
    
    # 创建训练器
    trainer = PPOTrainer(env, agent, train_config, logger)
    
    # 开始训练
    trainer.train()
    
if __name__ == "__main__":
    main() 