class SensorSuite:
    """传感器套件"""
    def __init__(self, config):
        self.sensors = {}
        
        # 相机配置
        self.cameras = {
            'front_rgb': {
                'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0, 'z': 1.4,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 800, 'height': 600, 'fov': 90
            },
            'bird_eye': {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 50.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 600, 'height': 600, 'fov': 90
            }
        }
        
        # 激光雷达配置
        self.lidar = {
            'type': 'sensor.lidar.ray_cast',
            'x': 0.0, 'y': 0.0, 'z': 2.4,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'range': 50,
            'channels': 32,
            'points_per_second': 56000,
            'upper_fov': 10,
            'lower_fov': -30,
            'rotation_frequency': 10
        } 