FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CARLA_ROOT=/opt/carla
ENV CARLA_VERSION=0.9.13

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    git \
    wget \
    curl \
    libpng16-16 \
    libtiff5 \
    libjpeg8 \
    libxerces-c3.2 \
    libxml2 \
    && rm -rf /var/lib/apt/lists/*

# 创建工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 下载CARLA
RUN mkdir -p ${CARLA_ROOT} \
    && wget -qO- https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_${CARLA_VERSION}.tar.gz \
    | tar -xzf - -C ${CARLA_ROOT}

# 设置Python路径
ENV PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py3.7-linux-x86_64.egg:${PYTHONPATH}

# 设置入口点
ENTRYPOINT ["python3", "-m"]
CMD ["src.main"] 