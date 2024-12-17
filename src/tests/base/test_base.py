from abc import ABC, abstractmethod
from typing import Dict, List
import carla
import numpy as np
import time
import logging

class BaseTest(ABC):
    """测试基类"""
    def __init__(self, client: carla.Client, config: Dict):
        self.client = client
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 测试结果
        self.results = {
            'success': False,
            'metrics': {},
            'errors': [],
            'duration': 0
        }
        
    @abstractmethod
    def setup(self):
        """测试准备"""
        pass
        
    @abstractmethod
    def run(self):
        """运行测试"""
        pass
        
    @abstractmethod
    def cleanup(self):
        """清理测试"""
        pass
        
    def execute(self) -> Dict:
        """执行测试"""
        start_time = time.time()
        
        try:
            self.setup()
            self.run()
            self.results['success'] = True
        except Exception as e:
            self.logger.error(f"Test failed: {str(e)}")
            self.results['errors'].append(str(e))
        finally:
            self.cleanup()
            
        self.results['duration'] = time.time() - start_time
        return self.results 