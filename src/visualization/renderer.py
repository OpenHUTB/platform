import pygame
import numpy as np
import cv2
from typing import Dict, List, Tuple
import carla

class Renderer:
    """可视化渲染器"""
    def __init__(self, width: int = 1280, height: int = 720):
        pygame.init()
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CARLA Visualization")
        
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        
    def render(self, sensor_data: Dict, info: Dict = None):
        """渲染画面"""
        # 清空屏幕
        self.display.fill((0, 0, 0))
        
        # 渲染相机视图
        if 'rgb' in sensor_data:
            self._render_camera(sensor_data['rgb'])
            
        # 渲染激光雷达
        if 'lidar' in sensor_data:
            self._render_lidar(sensor_data['lidar'])
            
        # 渲染车辆状态
        if info:
            self._render_vehicle_state(info)
            
        # 渲染调试信息
        if 'debug' in info:
            self._render_debug_info(info['debug'])
            
        # 更新显示
        pygame.display.flip()
        
    def _render_camera(self, image: np.ndarray):
        """渲染相机图像"""
        # 转换为pygame surface
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        
        # 显示在左侧
        self.display.blit(surface, (0, 0))
        
    def _render_lidar(self, points: np.ndarray):
        """渲染激光雷达点云"""
        # 生成BEV图像
        bev_image = self._generate_bev(points)
        
        # 转换为pygame surface
        surface = pygame.surfarray.make_surface(bev_image)
        
        # 显示在右上角
        self.display.blit(surface, (self.width//2, 0))
        
    def _render_vehicle_state(self, info: Dict):
        """渲染车辆状态"""
        # 创建文本
        font = pygame.font.Font(None, 36)
        
        # 显示速度
        speed_text = font.render(
            f"Speed: {info.get('speed', 0):.1f} km/h",
            True, (255, 255, 255)
        )
        self.display.blit(speed_text, (10, self.height - 100))
        
        # 显示转向角
        steer_text = font.render(
            f"Steer: {info.get('steer', 0):.2f}",
            True, (255, 255, 255)
        )
        self.display.blit(steer_text, (10, self.height - 60))
        
    def _render_debug_info(self, debug_info: Dict):
        """渲染调试信息"""
        font = pygame.font.Font(None, 24)
        y = 10
        
        for key, value in debug_info.items():
            text = font.render(f"{key}: {value}", True, (255, 255, 255))
            self.display.blit(text, (10, y))
            y += 25 
        
    def _render_trajectory(self, trajectory: List[carla.Location], color: Tuple[int, int, int] = (0, 255, 0)):
        """渲染轨迹"""
        if len(trajectory) < 2:
            return
        
        # 转换为屏幕坐标
        points = []
        for loc in trajectory:
            x = int(loc.x * self.scale + self.width/2)
            y = int(-loc.y * self.scale + self.height/2)
            points.append((x, y))
        
        # 绘制轨迹线
        pygame.draw.lines(self.display, color, False, points, 2)
        
    def _render_birdview(self, vehicle_state: Dict):
        """渲染鸟瞰图"""
        # 创建背景
        surface = pygame.Surface((400, 400))
        surface.fill((50, 50, 50))
        
        # 绘制道路
        for waypoint in vehicle_state['nearby_waypoints']:
            x = int(waypoint.transform.location.x * 2 + 200)
            y = int(-waypoint.transform.location.y * 2 + 200)
            pygame.draw.circle(surface, (200, 200, 200), (x, y), 2)
        
        # 绘制其他车辆
        for vehicle in vehicle_state['nearby_vehicles']:
            x = int(vehicle.get_location().x * 2 + 200)
            y = int(-vehicle.get_location().y * 2 + 200)
            pygame.draw.circle(surface, (255, 0, 0), (x, y), 4)
        
        # 绘制自车
        pygame.draw.circle(surface, (0, 255, 0), (200, 200), 6)
        
        # 显示在右下角
        self.display.blit(surface, (self.width-420, self.height-420))
        
    def _render_attention(self, attention_weights: np.ndarray, image: np.ndarray):
        """渲染注意力热图"""
        # 调整注意力图大小
        attention = cv2.resize(attention_weights, (image.shape[1], image.shape[0]))
        
        # 生成热图
        heatmap = cv2.applyColorMap(np.uint8(255*attention), cv2.COLORMAP_JET)
        
        # 叠加到原图
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        # 转换为pygame surface
        surface = pygame.surfarray.make_surface(overlay.swapaxes(0, 1))
        
        # 显示在右侧
        self.display.blit(surface, (self.width//2, self.height//2))
        
    def _generate_bev(self, points: np.ndarray) -> np.ndarray:
        """生成鸟瞰图"""
        # 创建画布
        bev_size = 400
        bev_range = 50.0
        bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        
        # 转换点云坐标
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # 过滤点云
        mask = (abs(x) < bev_range) & (abs(y) < bev_range)
        x = x[mask]
        y = y[mask]
        z = z[mask]
        
        # 转换为像素坐标
        x_img = ((x + bev_range) * bev_size / (2 * bev_range)).astype(np.int32)
        y_img = ((y + bev_range) * bev_size / (2 * bev_range)).astype(np.int32)
        
        # 根据高度着色
        colors = self._height_to_color(z)
        
        # 绘制点云
        for x, y, color in zip(x_img, y_img, colors):
            cv2.circle(bev, (x, y), 1, color, -1)
            
        return bev
        
    def _height_to_color(self, heights: np.ndarray) -> np.ndarray:
        """高度转颜色"""
        # 归一化高度
        min_h = heights.min()
        max_h = heights.max()
        norm_h = (heights - min_h) / (max_h - min_h + 1e-6)
        
        # 创建颜色映射
        colors = np.zeros((len(heights), 3))
        
        # 低处为蓝色，高处为红色
        colors[:, 0] = norm_h * 255  # R
        colors[:, 2] = (1 - norm_h) * 255  # B
        
        return colors.astype(np.uint8)
        
    def _render_prediction(self, predictions: Dict):
        """渲染预测结果"""
        # 创建文本
        font = pygame.font.Font(None, 32)
        y = self.height - 150
        
        # 显示动作预测
        if 'action' in predictions:
            action = predictions['action']
            text = font.render(
                f"Steer: {action[0]:.2f} Throttle: {action[1]:.2f} Brake: {action[2]:.2f}",
                True, (255, 255, 255)
            )
            self.display.blit(text, (self.width//2, y))
            y += 30
            
        # 显示值函数预测
        if 'value' in predictions:
            value = predictions['value']
            text = font.render(f"Value: {value:.2f}", True, (255, 255, 255))
            self.display.blit(text, (self.width//2, y))
            y += 30
            
        # 显示其他预测
        if 'aux' in predictions:
            for key, value in predictions['aux'].items():
                text = font.render(f"{key}: {value:.2f}", True, (255, 255, 255))
                self.display.blit(text, (self.width//2, y))
                y += 30