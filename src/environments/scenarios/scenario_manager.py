class ScenarioManager:
    """场景管理器"""
    def __init__(self, config):
        self.scenarios = {
            'urban': {
                'map': 'Town03',
                'weather': self._get_weather_presets(),
                'traffic': {
                    'vehicles': 30,
                    'pedestrians': 20,
                    'behaviors': ['normal', 'aggressive', 'cautious']
                },
                'events': [
                    'sudden_braking',
                    'pedestrian_crossing',
                    'vehicle_cutting_in'
                ]
            },
            'highway': {
                'map': 'Town04',
                'weather': self._get_weather_presets(),
                'traffic': {
                    'vehicles': 50,
                    'behaviors': ['normal', 'aggressive']
                },
                'events': [
                    'vehicle_cutting_in',
                    'slow_vehicle',
                    'traffic_jam'
                ]
            }
        } 