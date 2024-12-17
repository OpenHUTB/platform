# API参考文档

## 环境模块

### CarlaEnv
基础环境类，提供与CARLA交互的标准接口。

```python
class CarlaEnv:
    def __init__(self, config: Dict):
        """
        初始化环境
        
        参数:
            config (Dict): 环境配置
        """
        pass
        
    def reset(self) -> Dict:
        """
        重置环境
        
        返回:
            Dict: 初始观测
        """
        pass
        
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        环境步进
        
        参数:
            action (np.ndarray): 动作
            
        返回:
            Tuple[Dict, float, bool, Dict]: (观测, 奖励, 结束标志, 信息)
        """
        pass
```

### SensorManager
传感器管理器，处理所有传感器的创建、数据收集和清理。

```python
class SensorManager:
    def __init__(self, config: Dict):
        """
        初始化传感器管理器
        
        参数:
            config (Dict): 传感器配置
        """
        pass
```

## 算法模块

### BaseAlgorithm
算法基类，定义了算法的标准接口。

```python
class BaseAlgorithm:
    def __init__(self, config: Dict):
        """
        初始化算法
        
        参数:
            config (Dict): 算法配置
        """
        pass
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        预测动作
        
        参数:
            state (np.ndarray): 状态
            
        返回:
            np.ndarray: 动作
        """
        pass
```

## 训练模块

### TrainManager
训练管理器，负责整个训练流程的控制。

```python
class TrainManager:
    def __init__(self, config: Dict):
        """
        初始化训练管理器
        
        参数:
            config (Dict): 训练配置
        """
        pass
        
    def train(self):
        """开始训练"""
        pass
```

## 评估模块

### ModelEvaluator
模型评估器，提供全面的评估功能。

```python
class ModelEvaluator:
    def __init__(self, env, agent, config: Dict):
        """
        初始化评估器
        
        参数:
            env: 评估环境
            agent: 待评估的智能体
            config (Dict): 评估配置
        """
        pass
```

## 可视化模块

### Dashboard
训练监控仪表盘。

```python
class Dashboard:
    def __init__(self, config: Dict):
        """
        初始化仪表盘
        
        参数:
            config (Dict): 可视化配置
        """
        pass
``` 