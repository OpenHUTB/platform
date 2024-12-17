CARLA自动驾驶算法测试平台
=======================

.. toctree::
   :maxdepth: 2
   :caption: 目录:

   introduction
   installation
   quickstart
   tutorials/index
   api/index
   development/index
   faq

简介
----

CARLA自动驾驶算法测试平台是一个基于CARLA仿真器的自动驾驶算法开发和测试环境。
本平台提供了完整的工具链,支持算法开发、训练、测试和评估的全流程。

主要特性
-------

* 完整的CARLA环境配置和场景管理
* 集成化的数据收集和可视化系统
* 标准化的训练流程和评估体系
* 模块化的算法接口设计
* 完整的实验记录和日志系统
* 支持多人协作的项目结构

快速开始
-------

.. code-block:: bash

   # 克隆项目
   git clone https://github.com/your-org/carla-test-platform.git
   cd carla-test-platform

   # 安装依赖
   pip install -r requirements.txt

   # 运行示例
   python examples/basic_example.py 