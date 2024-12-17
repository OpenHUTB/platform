import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import cv2
import time

class Dashboard:
    """训练监控仪表盘"""
    def __init__(self, config: Dict):
        self.config = config
        self.setup_layout()
        self.metrics_history = {
            'rewards': [],
            'lengths': [],
            'success_rates': [],
            'collision_rates': [],
            'steps': []
        }
        
    def setup_layout(self):
        """设置页面布局"""
        st.set_page_config(layout="wide")
        st.title("CARLA自动驾驶训练监控")
        
        # 创建侧边栏
        self.setup_sidebar()
        
        # 创建主面板
        self.col1, self.col2 = st.columns(2)
        
        # 创建图表占位符
        with self.col1:
            self.reward_chart = st.empty()
            self.metrics_chart = st.empty()
            
        with self.col2:
            self.camera_view = st.empty()
            self.lidar_view = st.empty()
            
    def setup_sidebar(self):
        """设置侧边栏"""
        st.sidebar.title("训练控制")
        
        # 训练控制
        if st.sidebar.button("暂停训练"):
            self.pause_training()
            
        if st.sidebar.button("继续训练"):
            self.resume_training()
            
        # 评估设置
        st.sidebar.title("评估设置")
        self.eval_episodes = st.sidebar.slider("评估回合数", 5, 50, 10)
        
        # 可视化设置
        st.sidebar.title("可视化设置")
        self.show_camera = st.sidebar.checkbox("显示相机视图", True)
        self.show_lidar = st.sidebar.checkbox("显示激光雷达", True)
        
    def update(self, stats: Dict):
        """更新显示"""
        # 更新指标历史
        self.update_metrics(stats)
        
        # 更新图表
        self.update_charts()
        
        # 更新传感器视图
        if 'camera' in stats and self.show_camera:
            self.update_camera_view(stats['camera'])
            
        if 'lidar' in stats and self.show_lidar:
            self.update_lidar_view(stats['lidar'])
            
    def update_metrics(self, stats: Dict):
        """更新指标历史"""
        self.metrics_history['rewards'].append(stats['reward'])
        self.metrics_history['lengths'].append(stats['length'])
        self.metrics_history['success_rates'].append(stats.get('success_rate', 0))
        self.metrics_history['collision_rates'].append(stats.get('collision_rate', 0))
        self.metrics_history['steps'].append(stats['total_steps'])
        
    def update_charts(self):
        """更新图表"""
        with self.col1:
            # 奖励曲线
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                x=self.metrics_history['steps'],
                y=self.metrics_history['rewards'],
                mode='lines',
                name='Episode Reward'
            ))
            self.reward_chart.plotly_chart(fig_reward)
            
            # 性能指标
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Scatter(
                x=self.metrics_history['steps'],
                y=self.metrics_history['success_rates'],
                mode='lines',
                name='Success Rate'
            ))
            fig_metrics.add_trace(go.Scatter(
                x=self.metrics_history['steps'],
                y=self.metrics_history['collision_rates'],
                mode='lines',
                name='Collision Rate'
            ))
            self.metrics_chart.plotly_chart(fig_metrics)
            
    def update_camera_view(self, image: np.ndarray):
        """更新相机视图"""
        # 转换图像格式
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        if image.shape[0] == 3:  # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
            
        self.camera_view.image(image, caption="RGB Camera View")
        
    def update_lidar_view(self, points: np.ndarray):
        """更新激光雷达视图"""
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points[:, 3],
                colorscale='Viridis',
            )
        )])
        self.lidar_view.plotly_chart(fig)
        
    def pause_training(self):
        """暂停训练"""
        # TODO: 实现训练暂停逻辑
        pass
        
    def resume_training(self):
        """继续训练"""
        # TODO: 实现训练继续逻辑
        pass 