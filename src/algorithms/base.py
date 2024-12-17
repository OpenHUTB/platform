from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple
import torch
import numpy as np


class BaseAlgorithm(ABC):
    """算法基类，定义了算法需要实现的基本接口"""
    
    @abstractmethod
    def predict(self, observation: Dict[str, torch.Tensor]) -> np.ndarray:
        """根据观察预测动作"""
        raise NotImplementedError
        
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """更新算法"""
        raise NotImplementedError
        
    @abstractmethod
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """处理观察数据"""
        raise NotImplementedError
        
    @abstractmethod
    def process_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """处理经验数据"""
        raise NotImplementedError
        
    @abstractmethod
    def save(self, path: str) -> None:
        """保存模型"""
        raise NotImplementedError
        
    @abstractmethod
    def load(self, path: str) -> None:
        """加载模型"""
        raise NotImplementedError

class BaseNetwork(ABC, torch.nn.Module):
    """网络基类，定义了网络需要实现的基本接口"""
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """前向传播"""
        raise NotImplementedError
        
    @abstractmethod
    def reset_parameters(self) -> None:
        """重置参数"""
        raise NotImplementedError 