import numpy as np
from typing import Dict
import cv2


class ObservationProcessor:
    """观测处理器"""
    def __init__(self, config: Dict):
        self.config = config
        
        # 图像处理参数
        self.image_size = config.get('image_size', (84, 84))
        self.grayscale = config.get('grayscale', False)
        self.normalize = config.get('normalize', True)
        
        # 激光雷达处理参数
        self.max_points = config.get('max_points', 1000)
        self.max_range = config.get('max_range', 50.0)
        
        # 状态处理参数
        self.state_dims = config.get('state_dims', 10)
        
    def process(self, raw_obs: Dict) -> Dict[str, np.ndarray]:
        """处理原始观测"""
        processed_obs = {}
        
        # 处理相机图像
        if 'camera_rgb' in raw_obs:
            processed_obs['camera'] = self._process_image(raw_obs['camera_rgb'])
            
        # 处理激光雷达数据
        if 'lidar' in raw_obs:
            processed_obs['lidar'] = self._process_lidar(raw_obs['lidar'])
            
        # 处理状态信息
        if 'vehicle_state' in raw_obs:
            processed_obs['state'] = self._process_state(raw_obs['vehicle_state'])
            
        return processed_obs
        
    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """处理图像"""
        # 调整大小
        image = cv2.resize(image, self.image_size)
        
        # 转换为灰度图
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=0)
        else:
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            
        # 归一化
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            
        return image
        
    def _process_lidar(self, points: np.ndarray) -> np.ndarray:
        """处理激光雷达点云"""
        # 过滤超出范围的点
        mask = np.linalg.norm(points[:, :3], axis=1) < self.max_range
        points = points[mask]
        
        # 随机采样固定数量的点
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        elif len(points) < self.max_points:
            # 填充
            padding = np.zeros((self.max_points - len(points), points.shape[1]))
            points = np.concatenate([points, padding], axis=0)
            
        return points
        
    def _process_state(self, state: Dict) -> np.ndarray:
        """处理状态信息"""
        # 提取关键状态信息
        processed_state = np.zeros(self.state_dims)
        
        # 位置和方向
        processed_state[0:3] = state['position']
        processed_state[3:6] = state['rotation']
        
        # 速度
        processed_state[6:9] = state['velocity']
        
        # 其他状态信息...
        
        return processed_state 

class NavigationObservationProcessor(ObservationProcessor):
    """导航任务观测处理器"""
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 图像处理参数
        self.image_size = config.get('image_size', (84, 84))
        self.use_grayscale = config.get('use_grayscale', True)
        
        # 激光雷达处理参数
        self.lidar_range = config.get('lidar_range', 50.0)
        self.num_lidar_bins = config.get('num_lidar_bins', 64)
        
    def process(self, obs: Dict) -> Dict:
        """处理观测"""
        processed_obs = {}
        
        # 1. 处理相机图像
        if 'camera_rgb' in obs:
            processed_obs['camera'] = self._process_image(obs['camera_rgb'])
            
        # 2. 处理激光雷达数据
        if 'lidar' in obs:
            processed_obs['lidar'] = self._process_lidar(obs['lidar'])
            
        # 3. 处理车辆状态
        if 'vehicle_state' in obs:
            processed_obs['vehicle_state'] = self._process_vehicle_state(obs['vehicle_state'])
            
        # 4. 处理导航信息
        if 'navigation' in obs:
            processed_obs['navigation'] = self._process_navigation(obs['navigation'])
            
        return processed_obs
        
    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """处理图像"""
        # 调整大小
        image = cv2.resize(image, self.image_size)
        
        # 转换为灰度图
        if self.use_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.expand_dims(image, axis=0)  # 添加通道维度
        else:
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        return image
        
    def _process_lidar(self, points: np.ndarray) -> np.ndarray:
        """处理激光雷达数据"""
        # 过滤超出范围的点
        mask = np.linalg.norm(points[:, :2], axis=1) < self.lidar_range
        points = points[mask]
        
        # 转换为极坐标
        angles = np.arctan2(points[:, 1], points[:, 0])
        distances = np.linalg.norm(points[:, :2], axis=1)
        
        # 创建距离直方图
        hist, _ = np.histogram(
            angles,
            bins=self.num_lidar_bins,
            range=(-np.pi, np.pi),
            weights=distances
        )
        
        # 归一化
        hist = hist / self.lidar_range
        
        return hist
        
    def _process_vehicle_state(self, state: Dict) -> np.ndarray:
        """处理车辆状态"""
        # 提取关键状态信息
        processed_state = np.zeros(10)
        
        # 位置
        processed_state[0:3] = state['position']
        
        # 旋转
        processed_state[3:6] = state['rotation']
        
        # 速度
        processed_state[6:9] = state['velocity']
        
        # 总速度
        processed_state[9] = np.linalg.norm(state['velocity'])
        
        return processed_state
        
    def _process_navigation(self, nav_info: Dict) -> np.ndarray:
        """处理导航信息"""
        processed_nav = np.zeros(3)
        
        # 到目标的距离
        processed_nav[0] = nav_info['distance'] / 100.0  # 归一化到[0,1]
        
        # 到目标的角度
        processed_nav[1] = nav_info['angle'] / np.pi  # 归一化到[-1,1]
        
        # 完成进度
        processed_nav[2] = nav_info['progress']
        
        return processed_nav 