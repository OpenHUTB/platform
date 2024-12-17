# CARLA自动驾驶算法测试平台

## 简介
CARLA自动驾驶算法测试平台是一个基于CARLA仿真器的自动驾驶算法开发和测试环境。本平台提供了完整的工具链，支持算法开发、训练、测试和评估的全流程。

## 快速开始

### 1. 安装
```bash
# 克隆项目
git clone https://github.com/your-org/carla-test-platform.git
cd carla-test-platform

# 安装依赖
pip install -r requirements.txt

# 配置CARLA
python scripts/setup_dev_env.py
```

### 2. 运行示例
```bash
# 运行基础示例
python examples/basic_example.py

# 运行训练示例
python examples/train_example.py
```

## 主要功能

### 1. 环境系统
- 标准化的环境接口
- 完整的传感器支持
- 可配置的场景系统
- 灵活的奖励设计

### 2. 算法系统
- 模块化的算法框架
- 多种算法实现
- 参数配置系统
- 模型保存和加载

### 3. 训练系统
- 多进程训练支持
- 课程学习
- 经验回放
- 训练监控

### 4. 评估系统
- 多维度评估指标
- 分布式评估
- 性能基准测试
- 可视化分析

### 5. 可视化系统
- 实时训练监控
- 传感器数据显示
- 训练曲线绘制
- 评估结果展示

## 文档导航

- [安装指南](installation.md)
- [快速入门](quickstart.md)
- [环境配置](configuration.md)
- [算法开发](development/algorithm_development.md)
- [训练指南](guides/training_guide.md)
- [评估指南](guides/evaluation_guide.md)
- [API参考](api/index.md) 