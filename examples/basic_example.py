"""基础示例"""
import carla
import numpy as np
from src.environments import CarlaEnv
from src.algorithms.rl.ppo.ppo_agent import PPOAgent
from src.visualization.dashboard.dashboard_app import DashboardApp


def main():
    # 创建环境
    env_config = {
        'carla': {
            'host': 'localhost',
            'port': 2000,
            'timeout': 10.0
        },
        'sensors': {
            'camera_rgb': {
                'width': 800,
                'height': 600
            }
        }
    }
    
    env = CarlaEnv(env_config)
    
    # 创建智能体
    agent_config = {
        'obs_dim': 10,
        'act_dim': 2,
        'hidden_dim': 256,
        'learning_rate': 3e-4
    }
    
    agent = PPOAgent(agent_config)
    
    # 运行一个回合
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 选择动作
        action = agent.predict(obs)
        
        # 执行动作
        next_obs, reward, done, info = env.step(action)
        
        total_reward += reward
        obs = next_obs
        
    print(f"回合完成,总奖励: {total_reward}")


if __name__ == "__main__":
    main()
