import carla
from typing import Dict, Optional, List
import numpy as np
import yaml

class ScenarioManager:
    """场景管理器"""
    def __init__(self, client: carla.Client):
        self.client = client
        self.world = None
        self.current_scenario = None
        self.traffic_manager = None
        
    def load_scenario(self, config_path: str):
        """加载场景配置"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # 加载地图
        self.world = self.client.load_world(config['map'])
        
        # 设置天气
        weather = carla.WeatherParameters(**config['weather'])
        self.world.set_weather(weather)
        
        # 设置交通管理器
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        
        # 生成NPC车辆
        self._spawn_npcs(config.get('npcs', {}))
        
        # 生成行人
        self._spawn_pedestrians(config.get('pedestrians', {}))
        
        self.current_scenario = config
        
    def _spawn_npcs(self, npc_config: Dict):
        """生成NPC车辆"""
        number = npc_config.get('number', 50)
        spawn_points = self.world.get_map().get_spawn_points()
        
        # 批量生成车辆
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        batch = []
        for _ in range(number):
            blueprint = np.random.choice(blueprints)
            spawn_point = np.random.choice(spawn_points)
            batch.append(carla.command.SpawnActor(blueprint, spawn_point))
            
        # 执行批处理
        vehicles = self.client.apply_batch_sync(batch)
        
        # 设置自动驾驶
        for vehicle in vehicles:
            if vehicle.error:
                continue
            vehicle.set_autopilot(True, self.traffic_manager.get_port())
            
    def _spawn_pedestrians(self, ped_config: Dict):
        """生成行人"""
        number = ped_config.get('number', 30)
        spawn_points = self.world.get_random_location_from_navigation()
        
        # 生成行人和AI控制器
        blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        batch = []
        for _ in range(number):
            blueprint = np.random.choice(blueprints)
            spawn_point = carla.Transform(spawn_points)
            batch.append(carla.command.SpawnActor(blueprint, spawn_point))
            
        pedestrians = self.client.apply_batch_sync(batch)
        
        # 设置行人AI
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        batch = []
        for pedestrian in pedestrians:
            if pedestrian.error:
                continue
            batch.append(carla.command.SpawnActor(walker_controller_bp, 
                                                carla.Transform(),
                                                pedestrian.actor_id))
                                                
        controllers = self.client.apply_batch_sync(batch)
        
        # 启动行人AI
        for controller in controllers:
            if controller.error:
                continue
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            
    def reset_scenario(self):
        """重置当前场景"""
        if self.current_scenario:
            self.load_scenario(self.current_scenario)
            
    def cleanup(self):
        """清理场景"""
        if self.world:
            actors = self.world.get_actors()
            for actor in actors:
                actor.destroy() 