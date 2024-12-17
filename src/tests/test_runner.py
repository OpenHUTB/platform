from typing import Dict, List
import carla
import yaml
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from src.tests.scenarios.scenario_test import ScenarioTest
from src.tests.performance.performance_test import PerformanceTest

class TestRunner:
    """测试运行器"""
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 创建CARLA客户端
        self.client = carla.Client(
            self.config.get('host', 'localhost'),
            self.config.get('port', 2000)
        )
        self.client.set_timeout(10.0)
        
        # 测试结果目录
        self.results_dir = self.config.get('results_dir', 'test_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run_tests(self):
        """运行所有测试"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'tests': {}
        }
        
        # 运行场景测试
        if 'scenario_tests' in self.config:
            results['tests']['scenarios'] = self._run_scenario_tests()
            
        # 运行性能测试
        if 'performance_tests' in self.config:
            results['tests']['performance'] = self._run_performance_tests()
            
        # 保存结果
        self._save_results(results)
        
        return results
        
    def _run_scenario_tests(self) -> Dict:
        """运行场景测试"""
        scenario_results = {}
        
        # 并行运行测试
        with ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
            future_to_scenario = {
                executor.submit(self._run_single_scenario_test, scenario_config): name
                for name, scenario_config in self.config['scenario_tests'].items()
            }
            
            for future in future_to_scenario:
                name = future_to_scenario[future]
                try:
                    scenario_results[name] = future.result()
                except Exception as e:
                    scenario_results[name] = {
                        'success': False,
                        'error': str(e)
                    }
                    
        return scenario_results
        
    def _run_performance_tests(self) -> Dict:
        """���行性能测试"""
        performance_results = {}
        
        for name, test_config in self.config['performance_tests'].items():
            test = PerformanceTest(self.client, test_config)
            performance_results[name] = test.execute()
            
        return performance_results
        
    def _run_single_scenario_test(self, config: Dict) -> Dict:
        """运行单个场景测试"""
        test = ScenarioTest(self.client, config)
        return test.execute()
        
    def _save_results(self, results: Dict):
        """保存测试结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = os.path.join(self.results_dir, f'test_results_{timestamp}.json')
        
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4) 