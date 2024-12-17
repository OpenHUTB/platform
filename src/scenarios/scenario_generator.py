import carla
import numpy as np
from typing import Dict, List, Optional
import random

class ScenarioGenerator:
    """场景生成器"""
    def __init__(self, client: carla.Client, config: Dict):
        self.client = client
        self.config = config
        self.world = None
        self.traffic_manager = None
        
    def generate_scenario(self, scenario_type: str) -> Dict:
        """生成场景"""
        if scenario_type == 'urban':
            return self._generate_urban_scenario()
        elif scenario_type == 'highway':
            return self._generate_highway_scenario()
        elif scenario_type == 'intersection':
            return self._generate_intersection_scenario()
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
            
    def _generate_urban_scenario(self) -> Dict:
        """生成城市场景"""
        scenario = {
            'map': 'Town03',
            'weather': self._generate_weather(),
            'ego_vehicle': {
                'type': 'vehicle.tesla.model3',
                'spawn_point': self._get_random_spawn_point()
            },
            'npcs': {
                'vehicles': self._generate_npc_vehicles(20),
                'pedestrians': self._generate_pedestrians(10)
            },
            'events': self._generate_events('urban')
        }
        return scenario
        
    def _generate_highway_scenario(self) -> Dict:
        """生成高速公路场景"""
        scenario = {
            'map': 'Town04',
            'weather': self._generate_weather(),
            'ego_vehicle': {
                'type': 'vehicle.tesla.model3',
                'spawn_point': self._get_highway_spawn_point()
            },
            'npcs': {
                'vehicles': self._generate_npc_vehicles(30, 'highway'),
                'pedestrians': []
            },
            'events': self._generate_events('highway')
        }
        return scenario
        
    def _generate_intersection_scenario(self) -> Dict:
        """生成十字路口场景"""
        scenario = {
            'map': 'Town05',
            'weather': self._generate_weather(),
            'ego_vehicle': {
                'type': 'vehicle.tesla.model3',
                'spawn_point': self._get_intersection_spawn_point()
            },
            'npcs': {
                'vehicles': self._generate_npc_vehicles(15, 'intersection'),
                'pedestrians': self._generate_pedestrians(5)
            },
            'events': self._generate_events('intersection')
        }
        return scenario
        
    def _generate_weather(self) -> Dict:
        """生成天气条件"""
        return {
            'cloudiness': random.uniform(0, 100),
            'precipitation': random.uniform(0, 100),
            'precipitation_deposits': random.uniform(0, 100),
            'wind_intensity': random.uniform(0, 100),
            'sun_azimuth_angle': random.uniform(0, 360),
            'sun_altitude_angle': random.uniform(-90, 90)
        }
        
    def _generate_npc_vehicles(self, num_vehicles: int, scenario_type: str = 'urban') -> List[Dict]:
        """生成NPC车辆配置"""
        vehicles = []
        for _ in range(num_vehicles):
            vehicle = {
                'type': random.choice(['vehicle.audi.a2', 'vehicle.tesla.model3', 'vehicle.bmw.grandtourer']),
                'spawn_point': self._get_random_spawn_point(),
                'behavior': random.choice(['normal', 'aggressive', 'cautious']),
                'target_speed': self._get_target_speed(scenario_type)
            }
            vehicles.append(vehicle)
        return vehicles
        
    def _generate_pedestrians(self, num_pedestrians: int) -> List[Dict]:
        """生成行人配置"""
        pedestrians = []
        for _ in range(num_pedestrians):
            pedestrian = {
                'type': 'walker.pedestrian.0001',
                'spawn_point': self._get_random_sidewalk_point(),
                'behavior': random.choice(['standing', 'walking', 'running'])
            }
            pedestrians.append(pedestrian)
        return pedestrians
        
    def _generate_events(self, scenario_type: str) -> List[Dict]:
        """生成事件配置"""
        events = []
        if scenario_type == 'urban':
            events.extend([
                {'type': 'sudden_brake', 'probability': 0.3},
                {'type': 'pedestrian_crossing', 'probability': 0.4},
                {'type': 'vehicle_cutting_in', 'probability': 0.3}
            ])
        elif scenario_type == 'highway':
            events.extend([
                {'type': 'vehicle_cutting_in', 'probability': 0.4},
                {'type': 'slow_vehicle_ahead', 'probability': 0.3}
            ])
        elif scenario_type == 'intersection':
            events.extend([
                {'type': 'red_light_runner', 'probability': 0.2},
                {'type': 'crossing_vehicle', 'probability': 0.4},
                {'type': 'pedestrian_crossing', 'probability': 0.3}
            ])
        return events
        
    def _get_random_spawn_point(self) -> carla.Transform:
        """获取随机生成点"""
        spawn_points = self.world.get_map().get_spawn_points()
        return random.choice(spawn_points)
        
    def _get_highway_spawn_point(self) -> carla.Transform:
        """获取高速公路生成点"""
        # 根据地图特征选择合适的生成点
        spawn_points = self.world.get_map().get_spawn_points()
        # TODO: 实现高速公路生成点选择逻辑
        return random.choice(spawn_points)
        
    def _get_intersection_spawn_point(self) -> carla.Transform:
        """获取十字路口生成点"""
        # 根据地图特征选择合适的生成点
        spawn_points = self.world.get_map().get_spawn_points()
        # TODO: 实现十字路口生成点选择逻辑
        return random.choice(spawn_points)
        
    def _get_random_sidewalk_point(self) -> carla.Transform:
        """获取随机人行道点"""
        # TODO: 实现人行道点选择逻辑
        return self._get_random_spawn_point()
        
    def _get_target_speed(self, scenario_type: str) -> float:
        """获取目标速度"""
        if scenario_type == 'urban':
            return random.uniform(20, 40)  # km/h
        elif scenario_type == 'highway':
            return random.uniform(80, 120)  # km/h
        elif scenario_type == 'intersection':
            return random.uniform(20, 30)  # km/h
        return 30.0  # 默认速度 