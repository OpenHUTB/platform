# 安装指南

## 系统要求

- Ubuntu 20.04 或更高版本
- Python 3.7+
- CUDA 11.0+
- CARLA 0.9.13+
- 16GB+ RAM
- NVIDIA GPU (6GB+ VRAM)

## 基础安装

1. 安装系统依赖：
```bash
sudo apt-get update && sudo apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libpng16-16 \
    libtiff5 \
    libjpeg8
```

2. 安装CARLA：
```bash
# 下载CARLA
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz

# 解压到指定目录
mkdir -p /opt/carla
tar -xvzf CARLA_0.9.13.tar.gz -C /opt/carla

# 设置环境变量
echo 'export CARLA_ROOT=/opt/carla' >> ~/.bashrc
echo 'export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

3. 安装项目：
```bash
# 克隆项目
git clone https://github.com/your-org/carla-test-platform.git
cd carla-test-platform

# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e .
```

## 开发环境设置

1. 安装开发依赖：
```bash
pip install -e ".[dev]"
```

2. 设置pre-commit：
```bash
pre-commit install
```

3. 设置开发环境：
```bash
python scripts/setup_dev_env.py
```

## Docker安装

1. 安装Docker和NVIDIA Container Toolkit：
```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. 使用Docker Compose启动：
```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 常见问题

### CARLA安装问题

1. 如果遇到CARLA启动失败：
```bash
# 检查显卡驱动
nvidia-smi

# 检查CARLA进程
ps aux | grep CarlaUE4

# 检查端口占用
netstat -tulpn | grep 2000
```

2. 如果遇到Python API导入错误：
```bash
# 检查PYTHONPATH
echo $PYTHONPATH

# 检查egg文件是否存在
ls $CARLA_ROOT/PythonAPI/carla/dist/
```

### CUDA问题

1. 检查CUDA安装：
```bash
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

2. 如果CUDA版本不匹配：
```bash
# 安装特定版本的PyTorch
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## 更多资源

- [CARLA文档](https://carla.readthedocs.io/)
- [项目文档](docs/README.md)
- [API参考](docs/api/index.md)
- [常见问题](docs/faq.md) 