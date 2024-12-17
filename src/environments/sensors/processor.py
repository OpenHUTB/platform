import numpy as np
import cv2
from typing import Dict, List
import torch

class SensorProcessor:
    """传感器数据处理器"""
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        
        # 图像预处理参数
        self.img_size = config.get('img_size', (224, 224))
        self.normalize = config.get('normalize', True)
        
        # 激光雷达预处理参数
        self.lidar_range = config.get('lidar_range', 50.0)
        self.lidar_bins = config.get('lidar_bins', 32)
        
    def process_camera(self, image: np.ndarray) -> torch.Tensor:
        """处理相机图像"""
        # 调整大小
        image = cv2.resize(image, self.img_size)
        
        # 标准化
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image.to(self.device)
        
    def process_lidar(self, points: np.ndarray) -> torch.Tensor:
        """处理激光雷达点云"""
        # 过滤点云
        mask = np.linalg.norm(points[:, :2], axis=1) <= self.lidar_range
        points = points[mask]
        
        # 生成BEV图像
        bev = self._points_to_bev(points)
        
        # 转换为tensor
        bev = torch.from_numpy(bev).float()
        return bev.to(self.device)
        
    def process_all(self, sensor_data: Dict) -> Dict:
        """处理所有传感器数据"""
        processed = {}
        
        # 处理相机数据
        if 'rgb' in sensor_data:
            processed['rgb'] = self.process_camera(sensor_data['rgb'])
            
        # 处理激光雷达数据
        if 'lidar' in sensor_data:
            processed['lidar'] = self.process_lidar(sensor_data['lidar'])
            
        # 处理其他传感器数据
        if 'gnss' in sensor_data:
            processed['gnss'] = torch.tensor(sensor_data['gnss']).to(self.device)
            
        if 'imu' in sensor_data:
            processed['imu'] = torch.tensor(sensor_data['imu']).to(self.device)
            
        return processed
        
    def _points_to_bev(self, points: np.ndarray) -> np.ndarray:
        """点云转BEV图像"""
        # 创建高度图
        height_map = np.zeros((self.lidar_bins, self.lidar_bins))
        
        # 计算网格索引
        x_bins = np.int32((points[:, 0] + self.lidar_range) * 
                         self.lidar_bins / (2 * self.lidar_range))
        y_bins = np.int32((points[:, 1] + self.lidar_range) * 
                         self.lidar_bins / (2 * self.lidar_range))
        
        # 限制索引范围
        x_bins = np.clip(x_bins, 0, self.lidar_bins - 1)
        y_bins = np.clip(y_bins, 0, self.lidar_bins - 1)
        
        # 更新高度图
        for x, y, z in zip(x_bins, y_bins, points[:, 2]):
            height_map[y, x] = max(height_map[y, x], z)
            
        return height_map 

    def process_semantic(self, semantic_image: np.ndarray) -> Dict[str, torch.Tensor]:
        """处理语义分割图像"""
        # 调整大小
        semantic = cv2.resize(semantic_image, self.img_size, 
                             interpolation=cv2.INTER_NEAREST)
        
        # 提取不同类别的掩码
        masks = {}
        masks['road'] = (semantic == 7).astype(np.float32)
        masks['vehicle'] = (semantic == 10).astype(np.float32)
        masks['pedestrian'] = (semantic == 4).astype(np.float32)
        masks['traffic_sign'] = (semantic == 12).astype(np.float32)
        
        # 转换为tensor
        for k, v in masks.items():
            masks[k] = torch.from_numpy(v).to(self.device)
            
        return masks

    def process_depth(self, depth_image: np.ndarray) -> torch.Tensor:
        """处理深度图像"""
        # 调整大小
        depth = cv2.resize(depth_image, self.img_size)
        
        # 转换为实际深度值
        depth = depth.astype(np.float32)
        depth = 1000 * depth  # 转换为米
        
        # 裁剪范围
        depth = np.clip(depth, 0, 100.0)
        
        # 标准化
        depth = (depth - depth.mean()) / (depth.std() + 1e-7)
        
        # 转换为tensor
        depth = torch.from_numpy(depth).unsqueeze(0)
        return depth.to(self.device)

    def fuse_sensors(self, sensor_data: Dict) -> torch.Tensor:
        """融合多个传感器数据"""
        features = []
        
        # 处理RGB图像
        if 'rgb' in sensor_data:
            rgb_features = self.process_camera(sensor_data['rgb'])
            features.append(rgb_features)
            
        # 处理深度图像
        if 'depth' in sensor_data:
            depth_features = self.process_depth(sensor_data['depth'])
            features.append(depth_features)
            
        # 处理语义分割
        if 'semantic' in sensor_data:
            semantic_features = self.process_semantic(sensor_data['semantic'])
            features.extend(semantic_features.values())
            
        # 处理激光雷达
        if 'lidar' in sensor_data:
            lidar_features = self.process_lidar(sensor_data['lidar'])
            features.append(lidar_features)
            
        # 拼接特征
        return torch.cat(features, dim=1)