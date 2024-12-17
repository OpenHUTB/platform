# 配置指南

## 配置文件结构

项目使用YAML格式的配置文件，主要包括以下几个部分：

### 1. 环境配置
```yaml
environment:
  # CARLA连接设置
  carla:
    host: "localhost"
    port: 2000
    timeout: 10.0
    
  # 传感器设置
  sensors:
    camera_rgb:
      type: "sensor.camera.rgb"
      width: 800
      height: 600
      fov: 90
      
    lidar:
      type: "sensor.lidar.ray_cast"
      channels: 32
      range: 50.0
      points_per_second: 56000
      
  # 场景设置
  scenario:
    type: "urban"
    traffic_density: 50
    weather: "clear_noon"
```

### 2. 算法配置
```yaml
algorithm:
  # 网络设置
  network:
    type: "mlp"
    hidden_sizes: [256, 256]
    activation: "relu"
    
  # 训练设置
  training:
    learning_rate: 3.0e-4
    batch_size: 64
    n_epochs: 10
    
  # 其他设置
  gamma: 0.99
  gae_lambda: 0.95
```

### 3. 训练配置
```yaml
training:
  # 基础设置
  max_steps: 1000000
  save_interval: 10000
  eval_interval: 5000
  
  # 课程学习设置
  curriculum:
    enabled: true
    stages: ["basic", "medium", "hard"]
```

### 4. 评估配置
```yaml
evaluation:
  # 评估设置
  n_episodes: 100
  metrics: ["success_rate", "collision_rate"]
  
  # 场景设置
  scenarios:
    - type: "urban"
      difficulty: "easy"
    - type: "highway"
      difficulty: "medium"
```

### 5. 可视化配置
```yaml
visualization:
  # 仪表盘设置
  dashboard:
    enabled: true
    refresh_rate: 0.1
    
  # 记录设置
  recording:
    enabled: true
    save_video: true
    save_metrics: true
```

## 配置最佳实践

### 1. 环境配置
- 根据硬件性能调整传感器分辨率
- 合理设置传感器更新频率
- 适当配置场景复杂度

### 2. 算法配置
- 从小规模网络开始调试
- 使用合理的学习率范围
- 注意梯度裁剪设置

### 3. 训练配置
- 设置合适的评估间隔
- 使用课程学习加速训练
- 及时保存检查点

### 4. 评估配置
- 使用多样的测试场景
- 设置充分的评估回合数
- 选择合适的评估指标

### 5. 可视化配置
- 权衡刷新率和性能
- 合理配置存储空间
- 选择关键指标显示 