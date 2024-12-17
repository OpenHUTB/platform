import logging
import wandb
from typing import Dict, Any
from pathlib import Path
import json
import time


class Logger:
    """训练日志记录器"""
    def __init__(self, config: Dict):
        self.config = config
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'train.log'),
                logging.StreamHandler()
            ]
        )
        
        # 初始化wandb
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('project_name', 'carla-rl'),
                config=config,
                name=config.get('exp_name', None)
            )
            
        # 记录配置
        self._save_config(config)
        
        # 初始化指标
        self.metrics = {}
        self.step = 0
        
    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """记录指标"""
        if step is not None:
            self.step = step
            
        # 更新指标
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v)
            
        # 记录到wandb
        if self.config.get('use_wandb', False):
            wandb.log(metrics, step=self.step)
            
        # 打印到控制台
        logging.info(f"Step {self.step}: {metrics}")
        
    def log_model(self, model_path: str, metrics: Dict[str, float] = None):
        """记录模型"""
        if self.config.get('use_wandb', False):
            artifact = wandb.Artifact(
                name=f"model-{self.step}",
                type='model',
                metadata=metrics
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
    def save_metrics(self):
        """保存指标"""
        metrics_path = self.log_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def _save_config(self, config: Dict):
        """保存配置"""
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2) 