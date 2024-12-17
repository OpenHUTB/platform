# 环境开发指南

## 概述

本指南介绍如何在平台中开发自定义环境。平台基于CARLA模拟器，提供了灵活的接口来定义不同的自动驾驶任务。

## 基本步骤

1. 继承基类
2. 定义观察和动作空间
3. 实现环境逻辑
4. 配置环境
5. 测试环境

## 详细说明

### 1. 继承基类

所有自定义环境都需要继承`BaseEnv`类：

```python
from src.environments.base import BaseEnv

class MyEnvironment(BaseEnv):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        # 初始化环境
```

### 2. 定义空间

定义观察空间和动作空间：

```python
def get_observation_space(self) -> Dict[str, gym.Space]:
    return {
        'rgb': gym.spaces.Box(
            low=0, high=255,
            shape=(3, 224, 224),
            dtype=np.uint8
        ),
        'lidar': gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(32, 1024, 3),
            dtype=np.float32
        )
    }
    
def get_action_space(self) -> gym.Space:
    return gym.spaces.Box(
        low=np.array([-1.0, -1.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        dtype=np.float32
    )
```

### 3. 实现环境逻辑

实现必要的环境接口：

```python
def reset(self) -> Dict[str, Any]:
    """重置环境"""
    # 重置CARLA环境
    # 生成初始状态
    # 返回观察
    
def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
    """环境步进"""
    # 执行动作
    # 更新环境
    # 计算奖励
    # 检查终止条件
    # 返回结果
```

### 4. 配置环境

创建环境配置文件：

```yaml
# configs/environments/my_env.yaml
environment:
  name: my_environment
  version: 1.0
  
  # CARLA设置
  carla:
    host: localhost
    port: 2000
    timeout: 20.0
    
  # 传感器设置
  sensors:
    rgb_camera:
      enabled: true
      width: 800
      height: 600
      fov: 90
      
  # 任务设置
  task:
    max_steps: 1000
    target_speed: 30.0
    # ...
```

## 最佳实践

### 1. 传感器配置

- 合理设置传感器位置
- 优化采样频率
- 处理数据同步

### 2. 奖励设计

- 分解奖励函数
- 平衡各项权重
- 避免奖励稀疏

### 3. 场景生成

- 随机化环境
- 渐进式难度
- 覆盖边界情况

### 4. 性能优化

- 使用同步模式
- 优化渲染设置
- ���现并行环境

## 常见问题

### 1. 环境不稳定

可能的原因：
- CARLA服务器问题
- 传感器数据丢失
- 物理模拟异常

解决方案：
- 检查CARLA状态
- 实现数据缓冲
- 添加异常处理

### 2. 性能问题

可能的原因：
- 渲染开销大
- 传感器数据处理慢
- 物理计算密集

解决方案：
- 调整渲染质量
- 优化数据处理
- 使用多进程

## API参考

### BaseEnv

基础环境接口：

```python
class BaseEnv(gym.Env, ABC):
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        pass
        
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """环境步进"""
        pass
        
    # ...其他接口
```

### Registry

环境注册工具：

```python
from src.utils.registry import register_env, create_env

# 注册环境
@register_env("my_environment")
class MyEnvironment(BaseEnv):
    pass
    
# 创建环境实例
env = create_env("my_environment", config)
``` 