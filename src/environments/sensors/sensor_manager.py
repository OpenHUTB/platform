import carla
import numpy as np
from typing import Dict, Any
import queue
import weakref
import cv2


class SensorManager:
    """传感器管理器"""
    def __init__(self, config: Dict):
        self.config = config
        self.sensors = {}
        self.sensor_queues = {}
        
    def setup_sensors(self, world: carla.World, vehicle: carla.Vehicle):
        """设置传感器"""
        # 清理现有传感器
        self.cleanup()
        
        # 设置相机
        if 'cameras' in self.config:
            for name, cam_config in self.config['cameras'].items():
                self._setup_camera(world, vehicle, name, cam_config)
                
        # 设置激光雷达
        if 'lidars' in self.config:
            for name, lidar_config in self.config['lidars'].items():
                self._setup_lidar(world, vehicle, name, lidar_config)
                
        # 设置碰撞传感器
        if self.config.get('collision_sensor', True):
            self._setup_collision_sensor(world, vehicle)
            
        # 设置车道传感器
        if self.config.get('lane_sensor', True):
            self._setup_lane_sensor(world, vehicle)
            
    def get_sensor_data(self) -> Dict:
        """获取传感器数据"""
        sensor_data = {}
        
        # 获取相机数据
        for name, queue in self.sensor_queues.items():
            if not queue.empty():
                sensor_data[name] = queue.get()
                
        return sensor_data
        
    def cleanup(self):
        """清理传感器"""
        for sensor in self.sensors.values():
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        self.sensors.clear()
        self.sensor_queues.clear()
        
    def _setup_camera(self, world: carla.World, vehicle: carla.Vehicle, name: str, config: Dict):
        """设置相机"""
        # 创建相机蓝图
        blueprint = world.get_blueprint_library().find(config['type'])
        blueprint.set_attribute('image_size_x', str(config['width']))
        blueprint.set_attribute('image_size_y', str(config['height']))
        blueprint.set_attribute('fov', str(config['fov']))
        
        # 设置相机位置
        transform = carla.Transform(
            carla.Location(*config['position']),
            carla.Rotation(*config.get('rotation', [0, 0, 0]))
        )
        
        # 创建相机
        camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        
        # 设置回调
        queue = queue.Queue()
        camera.listen(lambda image: self._camera_callback(image, queue))
        
        self.sensors[name] = camera
        self.sensor_queues[name] = queue
        
    def _setup_lidar(self, world: carla.World, vehicle: carla.Vehicle, name: str, config: Dict):
        """设置激光雷达"""
        # 创建激光雷达蓝图
        blueprint = world.get_blueprint_library().find(config['type'])
        blueprint.set_attribute('channels', str(config['channels']))
        blueprint.set_attribute('range', str(config['range']))
        blueprint.set_attribute('points_per_second', str(config['points_per_second']))
        blueprint.set_attribute('rotation_frequency', str(config['rotation_frequency']))
        blueprint.set_attribute('upper_fov', str(config['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config['lower_fov']))
        
        # 设置激光雷达位置
        transform = carla.Transform(
            carla.Location(*config['position']),
            carla.Rotation(*config.get('rotation', [0, 0, 0]))
        )
        
        # 创建激光雷达
        lidar = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        
        # 设置回调
        queue = queue.Queue()
        lidar.listen(lambda data: self._lidar_callback(data, queue))
        
        self.sensors[name] = lidar
        self.sensor_queues[name] = queue
        
    def _setup_collision_sensor(self, world: carla.World, vehicle: carla.Vehicle):
        """设置碰撞传感器"""
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        
        queue = queue.Queue()
        sensor.listen(lambda event: self._collision_callback(event, queue))
        
        self.sensors['collision'] = sensor
        self.sensor_queues['collision'] = queue
        
    def _setup_lane_sensor(self, world: carla.World, vehicle: carla.Vehicle):
        """设置车道传感器"""
        blueprint = world.get_blueprint_library().find('sensor.other.lane_invasion')
        sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        
        queue = queue.Queue()
        sensor.listen(lambda event: self._lane_callback(event, queue))
        
        self.sensors['lane'] = sensor
        self.sensor_queues['lane'] = queue
        
    def _camera_callback(self, image, sensor_queue):
        """相机回调"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # 去除alpha通道
        sensor_queue.put(array)
        
    def _lidar_callback(self, data, sensor_queue):
        """激光雷达回调"""
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 4])
        sensor_queue.put(points)
        
    def _collision_callback(self, event, sensor_queue):
        """碰撞回调"""
        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        sensor_queue.put(intensity)
        
    def _lane_callback(self, event, sensor_queue):
        """车道偏离回调"""
        sensor_queue.put(True) 