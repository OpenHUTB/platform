"""仪表盘组件"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import cv2
from typing import Dict, List

class MetricsPanel:
    """指标面板"""
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_history = {
            metric['name']: [] for metric in config['metrics']
        }
        self.steps = []
        
    def update(self, metrics: Dict):
        """更新指标"""
        self.steps.append(len(self.steps))
        for name in self.metrics_history:
            if name in metrics:
                self.metrics_history[name].append(metrics[name])
                
    def render(self):
        """渲染面板"""
        # 创建图表
        fig = go.Figure()
        
        # 添加每个指标的曲线
        for metric in self.config['metrics']:
            name = metric['name']
            fig.add_trace(go.Scatter(
                x=self.steps,
                y=self.metrics_history[name],
                name=metric['title'],
                line=dict(color=metric['color'])
            ))
            
        # 更新布局
        fig.update_layout(
            title="训练指标",
            xaxis_title="步数",
            yaxis_title="值",
            height=self.config.get('height', 400)
        )
        
        st.plotly_chart(fig, use_container_width=True)

class SensorPanel:
    """传感器面板"""
    def __init__(self, config: Dict):
        self.config = config
        self.current_sensor = config['types'][0]
        
    def update(self, sensor_data: Dict):
        """更新传感器数据"""
        self.sensor_data = sensor_data
        
    def render(self):
        """渲染面板"""
        # 传感器选择
        self.current_sensor = st.selectbox(
            "选择传感器",
            self.config['types']
        )
        
        # 显示传感器数据
        if self.current_sensor in self.sensor_data:
            data = self.sensor_data[self.current_sensor]
            
            if 'camera' in self.current_sensor:
                self._render_camera(data)
            elif 'lidar' in self.current_sensor:
                self._render_lidar(data)
                
    def _render_camera(self, image: np.ndarray):
        """渲染相机图像"""
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        st.image(image, caption=self.current_sensor)
        
    def _render_lidar(self, points: np.ndarray):
        """渲染激光雷达点云"""
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
        
        fig.update_layout(
            title="激光雷达点云",
            height=self.config.get('height', 600)
        )
        
        st.plotly_chart(fig, use_container_width=True)

class VehicleStatePanel:
    """车辆状态面板"""
    def __init__(self, config: Dict):
        self.config = config
        
    def update(self, state: Dict):
        """更新车辆状态"""
        self.state = state
        
    def render(self):
        """渲染面板"""
        # 创建仪表盘
        fig = go.Figure()
        
        # 添加每个指标的仪表
        for i, metric in enumerate(self.config['metrics']):
            name = metric['name']
            if name in self.state:
                value = self.state[name]
                
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': f"{metric['title']} ({metric['unit']})"},
                    domain={'row': i//2, 'column': i%2},
                    gauge={'axis': {'range': [None, 100]}}
                ))
                
        # 更新布局
        fig.update_layout(
            grid={'rows': 2, 'columns': 2},
            height=self.config.get('height', 400)
        )
        
        st.plotly_chart(fig, use_container_width=True)

class TrainingCurvesPanel:
    """训练曲线面板"""
    def __init__(self, config: Dict):
        self.config = config
        self.history = {
            curve['name']: [] for curve in config['curves']
        }
        self.steps = []
        
    def update(self, metrics: Dict):
        """更新指标"""
        self.steps.append(len(self.steps))
        for name in self.history:
            if name in metrics:
                self.history[name].append(metrics[name])
                
    def render(self):
        """渲染面板"""
        # 创建图表
        fig = go.Figure()
        
        # 添加每条曲线
        for curve in self.config['curves']:
            name = curve['name']
            window_size = curve['window_size']
            
            # 计算移动平均
            values = np.array(self.history[name])
            if len(values) > window_size:
                values = np.convolve(
                    values,
                    np.ones(window_size)/window_size,
                    mode='valid'
                )
                
            fig.add_trace(go.Scatter(
                x=self.steps[-len(values):],
                y=values,
                name=curve['title']
            ))
            
        # 更新布局
        fig.update_layout(
            title="训练曲线",
            xaxis_title="步数",
            yaxis_title="值",
            height=self.config.get('height', 400)
        )
        
        st.plotly_chart(fig, use_container_width=True) 