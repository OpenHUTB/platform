class Evaluator:
    """评估器"""
    def __init__(self, config):
        # 评估配置
        self.n_episodes = config.get('n_episodes', 100)
        self.metrics = config.get('metrics', [
            'success_rate',
            'collision_rate',
            'completion_time',
            'average_speed',
            'comfort_metrics'
        ])
        
        # 场景设置
        self.scenarios = config.get('scenarios', [
            'urban_easy',
            'urban_medium',
            'urban_hard',
            'highway_easy',
            'highway_hard'
        ])
        
        # 评估环境
        self.envs = self._create_eval_envs()
        
        # 指标计算器
        self.metric_calculator = MetricCalculator() 