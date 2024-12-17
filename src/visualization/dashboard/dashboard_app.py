import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import json
import os


class DashboardApp:
    """训练监控仪表盘"""
    def __init__(self):
        st.set_page_config(page_title="CARLA训练监控", layout="wide")
        self.metrics_cache = {}
        
    def run(self):
        """运行仪表盘"""
        st.title("CARLA自动驾驶训练监控")
        
        # 侧边栏 - 实验选择
        self._render_sidebar()
        
        # 主面板
        col1, col2 = st.columns(2)
        
        with col1:
            # 训练指标
            self._render_training_metrics()
            
        with col2:
            # 评估指标
            self._render_eval_metrics()
            
        # 传感器数据可视化
        self._render_sensor_visualization()
        
        # 场景回放
        self._render_scenario_replay()
        
    def _render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("实验配置")
        
        # 选择实���
        exp_dir = "experiments/logs"
        experiments = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
        selected_exp = st.sidebar.selectbox("选择实验", experiments)
        
        # 更新频率
        update_interval = st.sidebar.slider("更新间隔(秒)", 1, 60, 5)
        
        # 显示选项
        st.sidebar.checkbox("显示训练曲线", value=True)
        st.sidebar.checkbox("显示评估指标", value=True)
        st.sidebar.checkbox("显示传感器数据", value=True)
        st.sidebar.checkbox("显示场景回放", value=True)
        
    def _render_training_metrics(self):
        """渲染训练指标"""
        st.subheader("训练指标")
        
        # 奖励曲线
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Scatter(
            x=self.metrics_cache.get('steps', []),
            y=self.metrics_cache.get('rewards', []),
            mode='lines',
            name='Episode Reward'
        ))
        st.plotly_chart(fig_reward)
        
        # 损失曲线
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=self.metrics_cache.get('steps', []),
            y=self.metrics_cache.get('policy_loss', []),
            mode='lines',
            name='Policy Loss'
        ))
        fig_loss.add_trace(go.Scatter(
            x=self.metrics_cache.get('steps', []),
            y=self.metrics_cache.get('value_loss', []),
            mode='lines',
            name='Value Loss'
        ))
        st.plotly_chart(fig_loss)
        
    def _render_eval_metrics(self):
        """渲染评估指标"""
        st.subheader("评估指标")
        
        # 成功率
        success_rate = self.metrics_cache.get('success_rate', [])
        if success_rate:
            st.metric("成功率", f"{success_rate[-1]:.2%}")
            
        # 碰撞率
        collision_rate = self.metrics_cache.get('collision_rate', [])
        if collision_rate:
            st.metric("碰撞率", f"{collision_rate[-1]:.2%}")
            
        # 完成率
        completion_rate = self.metrics_cache.get('completion_rate', [])
        if completion_rate:
            st.metric("完成率", f"{completion_rate[-1]:.2%}")
            
        # 性能指标图表
        metrics_df = pd.DataFrame({
            'Metric': ['平均速度', '平均加速度', '平均抖动'],
            'Value': [
                self.metrics_cache.get('avg_speed', [0])[-1],
                self.metrics_cache.get('avg_acceleration', [0])[-1],
                self.metrics_cache.get('avg_jerk', [0])[-1]
            ]
        })
        fig = px.bar(metrics_df, x='Metric', y='Value')
        st.plotly_chart(fig)
        
    def _render_sensor_visualization(self):
        """渲染传感器数据可视化"""
        st.subheader("传感器数据")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RGB相机
            st.image(self.metrics_cache.get('camera_rgb', None),
                    caption="RGB相机",
                    use_column_width=True)
            
        with col2:
            # 激光雷达
            if 'lidar' in self.metrics_cache:
                fig = px.scatter_3d(
                    self.metrics_cache['lidar'],
                    x='x', y='y', z='z',
                    color='intensity'
                )
                st.plotly_chart(fig)
                
    def _render_scenario_replay(self):
        """渲染场景回放"""
        st.subheader("场景回放")
        
        # 回放控制
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.button("播放")
        with col2:
            st.button("暂停")
        with col3:
            st.slider("进度", 0, 100, 0)
            
        # 回放视图
        if 'replay_frame' in self.metrics_cache:
            st.image(self.metrics_cache['replay_frame'],
                    caption="场景回放",
                    use_column_width=True) 