class ScenarioGenerator:
    """场景生成器"""
    def __init__(self, config: Dict):
        self.config = config
        self.scenario_manager = ScenarioManager(config)
        
    def generate_scenario(self, scenario_type: str) -> Dict:
        """生成具体场景"""
        base_scenario = self.scenario_manager.scenarios[scenario_type]
        
        # 生成天气条件
        weather = self._generate_weather(base_scenario['weather'])
        
        # 生成交通参与者
        traffic = self._generate_traffic(base_scenario['traffic'])
        
        # 生成事件
        events = self._generate_events(base_scenario['events'])
        
        return {
            'map': base_scenario['map'],
            'weather': weather,
            'traffic': traffic,
            'events': events
        }
        
    def _generate_weather(self, weather_presets: List) -> Dict:
        """生成天气条件"""
        weather = random.choice(weather_presets)
        return {
            'cloudiness': random.uniform(weather['cloudiness'][0], weather['cloudiness'][1]),
            'precipitation': random.uniform(weather['precipitation'][0], weather['precipitation'][1]),
            'sun_altitude_angle': random.uniform(weather['sun_altitude'][0], weather['sun_altitude'][1])
        }
        
    def _generate_traffic(self, traffic_config: Dict) -> Dict:
        """生成交通参与者"""
        return {
            'vehicles': self._generate_vehicles(traffic_config),
            'pedestrians': self._generate_pedestrians(traffic_config),
            'behaviors': self._assign_behaviors(traffic_config)
        } 