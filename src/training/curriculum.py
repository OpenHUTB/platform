from typing import Dict, List
import numpy as np
import json
import os

class CurriculumStage:
    """课程阶段"""
    def __init__(self, config: Dict):
        self.name = config['name']
        self.scenarios = config['scenarios']
        self.traffic_density = config.get('traffic_density', 0.0)
        self.weather_conditions = config.get('weather_conditions', ['clear_noon'])
        self.success_criteria = config.get('success_criteria', {})
        self.completion_threshold = config.get('completion_threshold', 0.8)
        
    def is_completed(self, metrics: Dict) -> bool:
        """检查阶段是否完成"""
        for metric_name, threshold in self.success_criteria.items():
            if metric_name not in metrics or metrics[metric_name] < threshold:
                return False
        return True

class CurriculumManager:
    """课程管理器"""
    def __init__(self, config: Dict):
        self.config = config
        self.current_stage_index = 0
        self.stages = self._create_stages()
        self.history = []
        
    def _create_stages(self) -> List[CurriculumStage]:
        """创建课程阶段"""
        stages = []
        for stage_config in self.config['stages']:
            stages.append(CurriculumStage(stage_config))
        return stages
        
    def get_current_stage(self) -> CurriculumStage:
        """获取当前阶段"""
        return self.stages[self.current_stage_index]
        
    def update(self, metrics: Dict) -> bool:
        """更新课程状态"""
        current_stage = self.get_current_stage()
        
        # 记录历史
        self.history.append({
            'stage': current_stage.name,
            'metrics': metrics
        })
        
        # 检查是否完成当前阶段
        if current_stage.is_completed(metrics):
            if self.current_stage_index < len(self.stages) - 1:
                self.current_stage_index += 1
                return True
                
        return False
        
    def get_stage_config(self) -> Dict:
        """获取当前阶段配置"""
        stage = self.get_current_stage()
        return {
            'scenarios': stage.scenarios,
            'traffic_density': stage.traffic_density,
            'weather_conditions': stage.weather_conditions
        }
        
    def save_progress(self, path: str):
        """保存进度"""
        save_data = {
            'current_stage': self.current_stage_index,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=4)
            
    def load_progress(self, path: str):
        """加载进度"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.current_stage_index = data['current_stage']
            self.history = data['history'] 