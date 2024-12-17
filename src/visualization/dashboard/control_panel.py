"""训练控制面板"""
import streamlit as st
from typing import Dict, Callable
import numpy as np
import json
from pathlib import Path

class ControlPanel:
    """训练控制面板"""
    def __init__(self, config: Dict):
        self.config = config
        self._setup_state()
        
    def _setup_state(self):
        """初始化状态"""
        if 'training_state' not in st.session_state:
            st.session_state.training_state = {
                'running': False,
                'current_episode': 0,
                'total_steps': 0,
                'best_reward': float('-inf')
            }
            
    def render(self, callbacks: Dict[str, Callable]):
        """渲染控制面板"""
        st.sidebar.title("训练控制")
        
        # 1. 训练控制
        self._render_training_controls(callbacks)
        
        # 2. 环境设置
        self._render_environment_settings(callbacks)
        
        # 3. 算法参数
        self._render_algorithm_settings(callbacks)
        
        # 4. 可视化设置
        self._render_visualization_settings(callbacks)
        
        # 5. 调试工具
        self._render_debug_tools(callbacks)
        
    def _render_training_controls(self, callbacks: Dict[str, Callable]):
        """渲染训练控制"""
        st.sidebar.subheader("训练控制")
        
        # 开始/暂停训练
        if st.sidebar.button(
            "暂停训练" if st.session_state.training_state['running'] else "开始训练"
        ):
            st.session_state.training_state['running'] = not st.session_state.training_state['running']
            if 'toggle_training' in callbacks:
                callbacks['toggle_training'](st.session_state.training_state['running'])
                
        # 保存/加载模型
        col1, col2 = st.sidebar.columns(2)
        if col1.button("保存模型"):
            if 'save_model' in callbacks:
                callbacks['save_model']()
                
        if col2.button("加载模型"):
            if 'load_model' in callbacks:
                callbacks['load_model']()
                
        # 训练进度
        st.sidebar.progress(st.session_state.training_state['total_steps'] / self.config['max_steps'])
        st.sidebar.text(f"Episode: {st.session_state.training_state['current_episode']}")
        st.sidebar.text(f"Steps: {st.session_state.training_state['total_steps']}")
        st.sidebar.text(f"Best Reward: {st.session_state.training_state['best_reward']:.2f}")
        
    def _render_environment_settings(self, callbacks: Dict[str, Callable]):
        """渲染环境设置"""
        st.sidebar.subheader("环境设置")
        
        # 场景选择
        scenario = st.sidebar.selectbox(
            "场景",
            ["城市", "高速", "路口"],
            key="scenario"
        )
        
        # 天气设置
        weather = st.sidebar.selectbox(
            "天气",
            ["晴天", "雨天", "夜晚"],
            key="weather"
        )
        
        # 交通密度
        traffic_density = st.sidebar.slider(
            "交通密度",
            0.0, 1.0, 0.5,
            key="traffic_density"
        )
        
        if 'update_env_settings' in callbacks:
            callbacks['update_env_settings']({
                'scenario': scenario,
                'weather': weather,
                'traffic_density': traffic_density
            })
            
    def _render_algorithm_settings(self, callbacks: Dict[str, Callable]):
        """渲染算法参数设置"""
        st.sidebar.subheader("算法参数")
        
        # 学习率
        learning_rate = st.sidebar.number_input(
            "学习率",
            1e-6, 1.0, 3e-4,
            format="%.0e",
            key="learning_rate"
        )
        
        # Batch大小
        batch_size = st.sidebar.number_input(
            "Batch大小",
            1, 1024, 64,
            key="batch_size"
        )
        
        # 奖励权重
        st.sidebar.text("奖励权重")
        reward_weights = {}
        col1, col2 = st.sidebar.columns(2)
        with col1:
            reward_weights['safety'] = st.number_input("安全", 0.0, 2.0, 1.0, key="w_safety")
            reward_weights['comfort'] = st.number_input("舒适", 0.0, 2.0, 0.3, key="w_comfort")
        with col2:
            reward_weights['efficiency'] = st.number_input("效率", 0.0, 2.0, 0.5, key="w_efficiency")
            reward_weights['task'] = st.number_input("任务", 0.0, 2.0, 1.0, key="w_task")
            
        if 'update_algo_settings' in callbacks:
            callbacks['update_algo_settings']({
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'reward_weights': reward_weights
            })
            
    def _render_visualization_settings(self, callbacks: Dict[str, Callable]):
        """渲染可视化设置"""
        st.sidebar.subheader("可视化设置")
        
        # 显示选项
        show_camera = st.sidebar.checkbox("显示相机视图", True, key="show_camera")
        show_lidar = st.sidebar.checkbox("显示激光雷达", True, key="show_lidar")
        show_metrics = st.sidebar.checkbox("显示指标", True, key="show_metrics")
        
        # 刷新间隔
        refresh_rate = st.sidebar.slider(
            "刷新间隔(秒)",
            0.1, 2.0, 0.5,
            key="refresh_rate"
        )
        
        if 'update_vis_settings' in callbacks:
            callbacks['update_vis_settings']({
                'show_camera': show_camera,
                'show_lidar': show_lidar,
                'show_metrics': show_metrics,
                'refresh_rate': refresh_rate
            })
            
    def _render_debug_tools(self, callbacks: Dict[str, Callable]):
        """渲染调试工具"""
        st.sidebar.subheader("调试工具")
        
        # 手动控制
        if st.sidebar.checkbox("手动控制", False, key="manual_control"):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                steering = st.slider("转向", -1.0, 1.0, 0.0, key="steering")
            with col2:
                throttle = st.slider("油门", 0.0, 1.0, 0.0, key="throttle")
                brake = st.slider("刹车", 0.0, 1.0, 0.0, key="brake")
                
            if 'manual_control' in callbacks:
                callbacks['manual_control']({
                    'steering': steering,
                    'throttle': throttle,
                    'brake': brake
                })
                
        # 场景重置
        if st.sidebar.button("重置场景"):
            if 'reset_scene' in callbacks:
                callbacks['reset_scene']()
                
        # 保存回放
        if st.sidebar.button("保存回放"):
            if 'save_replay' in callbacks:
                callbacks['save_replay']()
                
        # 性能分析
        if st.sidebar.checkbox("性能分析", False, key="profiling"):
            if 'toggle_profiling' in callbacks:
                callbacks['toggle_profiling'](True)
                
    def update_state(self, new_state: Dict):
        """更新状态"""
        st.session_state.training_state.update(new_state) 