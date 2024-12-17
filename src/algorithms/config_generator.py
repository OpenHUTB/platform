"""算法配置生成器"""
from typing import Dict, Any, Optional
import yaml
import os

class AlgorithmConfigGenerator:
    """算法配置生成器"""
    def __init__(self, base_config_dir: str = 'configs/algo_configs'):
        self.base_config_dir = base_config_dir
        
    def generate_config(self, 
                       algorithm: str,
                       env_info: Dict,
                       custom_config: Optional[Dict] = None) -> Dict:
        """生成算法配置"""
        # 加载基础配置
        base_config = self._load_base_config(algorithm)
        
        # 更新环境相关配置
        config = self._update_env_config(base_config, env_info)
        
        # 合并自定义配置
        if custom_config:
            config.update(custom_config)
            
        return config
        
    def _load_base_config(self, algorithm: str) -> Dict:
        """加载基础配置"""
        config_path = os.path.join(self.base_config_dir, f'{algorithm}.yaml')
        if not os.path.exists(config_path):
            return {}
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _update_env_config(self, config: Dict, env_info: Dict) -> Dict:
        """更新环境相关配置"""
        config['obs_dim'] = env_info.get('obs_dim')
        config['act_dim'] = env_info.get('act_dim')
        config['action_space'] = env_info.get('action_space')
        config['observation_space'] = env_info.get('observation_space')
        return config
        
    def save_config(self, config: Dict, save_path: str):
        """保存配置"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False) 