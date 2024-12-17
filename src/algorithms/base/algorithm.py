from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch
import numpy as np

class BaseAlgorithm(ABC):
    """算法基类"""
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def predict(self, state: np.ndarray) -> np.ndarray:
        """预测动作"""
        pass
        
    @abstractmethod
    def update(self, batch: Dict) -> Dict:
        """更新模型"""
        pass
        
    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass
        
    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass 