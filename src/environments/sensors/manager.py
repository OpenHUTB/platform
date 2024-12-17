class SensorManager:
    """传感器管理器"""
    def __init__(self, config: Dict):
        self.config = config
        self.sensors = {}
        self.data_buffers = {}
        self.processor = SensorProcessor(config)
        
    def setup_sensors(self, world: carla.World, vehicle: carla.Vehicle):
        """设置传感器"""
        # 相机设置
        self._setup_cameras(world, vehicle)
        
        # 激光雷达设置
        self._setup_lidar(world, vehicle)
        
        # GNSS和IMU设置
        self._setup_gnss_imu(world, vehicle)
        
        # 碰撞和车道传感器
        self._setup_collision_lane(world, vehicle)
        
    def _setup_cameras(self, world: carla.World, vehicle: carla.Vehicle):
        """设置相机"""
        camera_configs = {
            'rgb_front': {
                'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0, 'z': 1.4,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 800, 'height': 600, 'fov': 90
            },
            'depth_front': {
                'type': 'sensor.camera.depth',
                'x': 2.0, 'y': 0.0, 'z': 1.4,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 800, 'height': 600, 'fov': 90
            },
            'semantic_front': {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 2.0, 'y': 0.0, 'z': 1.4,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 800, 'height': 600, 'fov': 90
            }
        }
        
        for name, config in camera_configs.items():
            if self.config.get(f'use_{name}', True):
                self._create_camera(name, config, world, vehicle)
                
    def _create_camera(self, name: str, config: Dict, 
                      world: carla.World, vehicle: carla.Vehicle):
        """创建相机"""
        # 创建蓝图
        blueprint = world.get_blueprint_library().find(config['type'])
        blueprint.set_attribute('image_size_x', str(config['width']))
        blueprint.set_attribute('image_size_y', str(config['height']))
        blueprint.set_attribute('fov', str(config['fov']))
        
        # 设置位置
        transform = carla.Transform(
            carla.Location(x=config['x'], y=config['y'], z=config['z']),
            carla.Rotation(roll=config['roll'], pitch=config['pitch'], yaw=config['yaw'])
        )
        
        # 创建传感器
        sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        
        # 设置回调
        sensor.listen(lambda image: self._on_camera_data(name, image))
        
        self.sensors[name] = sensor
        self.data_buffers[name] = None
        
    def _on_camera_data(self, name: str, image):
        """相机数据回调"""
        # 转换为numpy数组
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        
        # 处理特殊相机类型
        if 'depth' in name:
            array = array[:, :, 0]  # 只使用R通道
        elif 'semantic' in name:
            array = array[:, :, 2]  # 只使用B通道
            
        # 存储数据
        self.data_buffers[name] = array 
        
    def _setup_lidar(self, world: carla.World, vehicle: carla.Vehicle):
        """设置激光雷达"""
        # 创建激光雷达配置
        lidar_config = {
            'type': 'sensor.lidar.ray_cast',
            'x': 0.0, 'y': 0.0, 'z': 2.4,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'range': 50.0,
            'channels': 32,
            'points_per_second': 56000,
            'rotation_frequency': 10,
            'upper_fov': 10.0,
            'lower_fov': -30.0
        }
        
        if self.config.get('use_lidar', True):
            # 创建蓝图
            blueprint = world.get_blueprint_library().find(lidar_config['type'])
            
            # 设置属性
            blueprint.set_attribute('range', str(lidar_config['range']))
            blueprint.set_attribute('channels', str(lidar_config['channels']))
            blueprint.set_attribute('points_per_second', str(lidar_config['points_per_second']))
            blueprint.set_attribute('rotation_frequency', str(lidar_config['rotation_frequency']))
            blueprint.set_attribute('upper_fov', str(lidar_config['upper_fov']))
            blueprint.set_attribute('lower_fov', str(lidar_config['lower_fov']))
            
            # 设置位置
            transform = carla.Transform(
                carla.Location(x=lidar_config['x'], y=lidar_config['y'], z=lidar_config['z']),
                carla.Rotation(roll=lidar_config['roll'], pitch=lidar_config['pitch'], yaw=lidar_config['yaw'])
            )
            
            # 创建传感器
            sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
            
            # 设置回调
            sensor.listen(lambda data: self._on_lidar_data('lidar', data))
            
            self.sensors['lidar'] = sensor
            self.data_buffers['lidar'] = None

    def _setup_gnss_imu(self, world: carla.World, vehicle: carla.Vehicle):
        """设置GNSS和IMU"""
        # GNSS配置
        gnss_config = {
            'type': 'sensor.other.gnss',
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'noise_alt_bias': 0.0,
            'noise_lat_bias': 0.0,
            'noise_lon_bias': 0.0,
            'noise_alt_stddev': 0.0,
            'noise_lat_stddev': 0.0,
            'noise_lon_stddev': 0.0
        }
        
        # IMU配置
        imu_config = {
            'type': 'sensor.other.imu',
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'noise_accel_stddev_x': 0.0,
            'noise_accel_stddev_y': 0.0,
            'noise_accel_stddev_z': 0.0,
            'noise_gyro_stddev_x': 0.0,
            'noise_gyro_stddev_y': 0.0,
            'noise_gyro_stddev_z': 0.0
        }
        
        if self.config.get('use_gnss', True):
            self._create_gnss(gnss_config, world, vehicle)
        if self.config.get('use_imu', True):
            self._create_imu(imu_config, world, vehicle)

    def _setup_collision_lane(self, world: carla.World, vehicle: carla.Vehicle):
        """设置碰撞和车道传感器"""
        # 碰撞传感器
        if self.config.get('use_collision', True):
            blueprint = world.get_blueprint_library().find('sensor.other.collision')
            sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
            sensor.listen(lambda event: self._on_collision(event))
            self.sensors['collision'] = sensor
            
        # 车道入侵传感器
        if self.config.get('use_lane_invasion', True):
            blueprint = world.get_blueprint_library().find('sensor.other.lane_invasion')
            sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
            sensor.listen(lambda event: self._on_lane_invasion(event))
            self.sensors['lane_invasion'] = sensor

    def _on_lidar_data(self, name: str, data):
        """激光雷达数据回调"""
        # 转换为numpy数组
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
        
        # 存储数据
        self.data_buffers[name] = points

    def _on_gnss_data(self, data):
        """GNSS数据回调"""
        self.data_buffers['gnss'] = {
            'latitude': data.latitude,
            'longitude': data.longitude,
            'altitude': data.altitude
        }

    def _on_imu_data(self, data):
        """IMU数据回调"""
        self.data_buffers['imu'] = {
            'accelerometer': [data.accelerometer.x, data.accelerometer.y, data.accelerometer.z],
            'gyroscope': [data.gyroscope.x, data.gyroscope.y, data.gyroscope.z],
            'compass': data.compass
        }

    def _on_collision(self, event):
        """碰撞事件回调"""
        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.data_buffers['collision'] = {
            'intensity': intensity,
            'actor_id': event.other_actor.id
        }

    def _on_lane_invasion(self, event):
        """车道入侵事件回调"""
        self.data_buffers['lane_invasion'] = {
            'crossed_lane_markings': [marking.type for marking in event.crossed_lane_markings]
        }

    def get_sensor_data(self) -> Dict:
        """获取所有传感器数据"""
        # 处理传感器数据
        processed_data = {}
        
        # 处理相机数据
        for name, data in self.data_buffers.items():
            if 'rgb' in name or 'depth' in name or 'semantic' in name:
                if data is not None:
                    processed_data[name] = self.processor.process_camera(data)
                
        # 处理激光雷达数据
        if 'lidar' in self.data_buffers and self.data_buffers['lidar'] is not None:
            processed_data['lidar'] = self.processor.process_lidar(self.data_buffers['lidar'])
            
        # 处理其他传感器数据
        for name in ['gnss', 'imu', 'collision', 'lane_invasion']:
            if name in self.data_buffers and self.data_buffers[name] is not None:
                processed_data[name] = self.data_buffers[name]
                
        return processed_data

    def cleanup(self):
        """清理传感器"""
        for sensor in self.sensors.values():
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
        self.data_buffers.clear()