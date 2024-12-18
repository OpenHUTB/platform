# 基于 Carla 的算法测试平台

## 项目概述

本项目是一个基于 Carla 模拟器的自动驾驶算法测试平台，提供了完整的训练和评估框架，支持自定义算法、环境和任务。

### 主要特性

### 环境系统
- 🚗 标准化的Gym风格环境接口
- 🎯 可配置的多场景系统
- 📷 完整的传感器支持(相机、激光雷达、IMU等)
- 🌍 丰富的交通场景生成
- 🎮 灵活的动作空间设计

### 算法框架
- 🧠 模块化的算法实现
- 📊 统一的数据收集接口
- 🔄 多进程训练支持
- 💾 经验回放机制
- 📈 课程学习支持

### 评估系统
- 📋 多维度评估指标
- 🚦 标准化测试场景
- 📊 自动化评估流程
- 🔍 详细的性能分析
- 📝 自动报告生成

### 可视化系统
- 📊 实时训练监控
- 🎥 传感器数据可视化
- 📈 训练曲线绘制
- 📷 场景回放功能
- 🎛️ 交互式控制面板

## 安装指南

### 系统要求
- Python 3.7+
- CUDA 11.0+
- CARLA 0.9.13+
- 16GB+ RAM
- NVIDIA GPU (6GB+ VRAM)

### 基础安装
```bash
# 克隆项目
git clone https://github.com/OpenHUTB/platform.git
cd platform

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux
# 或者
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 2. 运行示例

```bash
# 启动CARLA服务器（Linux）
./CarlaUE4.sh -quality-level=Epic
# Windows
CarlaUE4.exe

# 运行训练
python scripts/train.py --config configs/training/default.yaml --exp-name demo

# 运行评估
python scripts/evaluate.py --config configs/evaluation/default.yaml
```

### 3. Docker部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

## 项目结构

```
carla-test-platform/
├── src/                    # 源代码
│   ├── algorithms/         # 算法实现
│   ├── environments/       # 环境定义
│   ├── training/          # 训练系统
│   └── visualization/     # 可视化工具
├── configs/                # 配置文件
├── scripts/                # 运行脚本
├── tests/                  # 测试代码
├── docs/                   # 文档
└── examples/              # 示例代码
```

## 使用指南

### 1. 自定义算法

参考[算法开发指南](docs/development/algorithm_development.md)了解如何：
- 创建新算法
- 配置网络结构
- 实现训练逻辑
- 注册和使用

### 2. 自定义环境

参考[环境开发指南](docs/development/environment_development.md)了解如何：
- 定义新环境
- 配置传感器
- 实现奖励函数
- 生成场景

### 3. 评估系统

使用内置的评估工具：
```bash
# 运行基准测试
python scripts/benchmark.py --algo sac --env navigation

# 生成评估报告
python scripts/generate_report.py --results-dir results/benchmark
```

### 4. 可视化工具

启动可视化界面：
```bash
# 实时监控
python scripts/visualize.py --config configs/visualization/default.yaml

# 回放数据
python scripts/replay.py --log-file logs/episode_001.pkl
```

## 配置说明

### 1. 算法配置

```yaml
# configs/algorithms/sac.yaml
algorithm:
  name: sac
  network:
    encoder: resnet18
    hidden_sizes: [256, 256]
  training:
    batch_size: 256
    learning_rate: 3e-4
```

### 2. 环境配置

```yaml
# configs/environments/navigation.yaml
environment:
  name: navigation
  sensors:
    rgb_camera:
      enabled: true
      width: 800
      height: 600
  task:
    max_steps: 1000
```

## 性能优化

### 1. 数据处理
- 使用数据预取
- 实现并行环境
- 优化传感器配置

## 详细项目结构
```
carla-test-platform/
├── configs/                    # 配置文件
│   ├── algorithms/            # 算法配置
│   ├── environments/          # 环境配置
│   ├── evaluation/           # 评估配置
│   ├── scenarios/            # 场景配置
│   └── visualization/        # 可视化配置
├── src/                       # 源代码
│   ├── environments/          # 环境实现
│   │   ├── carla_env.py      # 基础环境
│   │   ├── sensors/          # 传感器模块
│   │   ├── scenarios/        # 场景生成
│   │   └── tasks/            # 任务定义
│   ├── algorithms/            # 算法实现
│   │   ├── base.py           # 算法基类
│   │   ├── sac/              # SAC算法
│   │   ├── ppo/              # PPO算法
│   │   └── td3/              # TD3算法
│   ├── training/             # 训练系统
│   │   ├── trainer.py        # 训练器
│   │   ├── buffer.py         # 经验回放
│   │   └── optimizer.py      # 优化器
│   ├── evaluation/           # 评估系统
│   │   ├── evaluator.py      # 评估器
│   │   ├── metrics.py        # 评估指标
│   │   └── analyzer.py       # 分析工具
│   ├── visualization/        # 可视化系统
│   │   ├── dashboard/        # 仪表盘
│   │   ├── renderer/         # 渲染器
│   │   └── reporter/         # 报告生成
│   └── utils/                # 工具函数
│       ├── logger.py         # 日志工具
│       ├── config.py         # 配置工具
│       └── registry.py       # 注册器
├── scripts/                   # 脚本工具
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   ├── visualize.py          # 可视化脚本
│   └── profile.py            # 性能分析
├── examples/                  # 示例代码
│   ├── basic_example.py      # 基础示例
│   ├── custom_task.py        # 自定义任务
│   └── custom_algorithm.py   # 自定义算法
├── tests/                    # 测试代码
│   ├── environments/         # 环境测试
│   ├── algorithms/           # 算法测试
│   └── integration/          # 集成测试
└── docs/                     # 文档
    ├── installation.md       # 安装指南
    ├── quickstart.md         # 快速入门
    ├── configuration.md      # 配置说明
    ├── development/          # 开发指南
    ├── guides/               # 使用指南
    └── api/                  # API文档
```

### 3. 内存管理
- 及时清理缓存
- 使用数据生成器
- 控制回放缓冲区大小

## 常见问题

### 1. 安装问题

Q: CARLA安装失败
A: 检查系统要求，确保显卡驱动更新

Q: 依赖冲突
A: 使用虚拟环境，按指定版本安装

### 2. 运行问题

Q: CARLA服务器无响应
A: 检查端口占用，重启服务器

Q: 训练不稳定
A: 调整学习率，检查奖励设计

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目基于 [项目](https://github.com/siupal/) 进行开发，采用MIT许可证，详见[LICENSE](LICENSE)文件。

## 联系方式

- 问题反馈：[GitHub Issues](https://github.com/OpenHUTB/platform/issues)
- 邮件联系：whd@hutb.edu.cn
- 技术讨论：[Discussions](https://github.com/OpenHUTB/platform/discussions)