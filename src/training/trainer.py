class Trainer:
    """训练器"""
    def __init__(self, config):
        # 训练配置
        self.max_steps = config.get('max_steps', 1000000)
        self.eval_interval = config.get('eval_interval', 10000)
        self.save_interval = config.get('save_interval', 10000)
        
        # 环境设置
        self.env = self._create_env(config['environment'])
        self.eval_env = self._create_env(config['environment'])
        
        # 算法设置
        self.algorithm = self._create_algorithm(config['algorithm'])
        
        # 日志设置
        self.logger = Logger(config['logging'])
        
        # 训练状态
        self.current_step = 0
        self.episodes = 0
        self.best_reward = float('-inf') 