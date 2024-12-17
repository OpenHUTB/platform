import carla
import numpy as np
from typing import List, Tuple, Optional

class RoutePlanner:
    """路径规划器"""
    def __init__(self, world_map: carla.Map):
        self.map = world_map
        self.sampling_resolution = 1.0  # meters
        self.debug = False
        
    def plan_route(self, start_location: carla.Location, 
                  end_location: carla.Location) -> List[carla.Location]:
        """规划路径"""
        # 获取起点和终点的路点
        start_waypoint = self.map.get_waypoint(start_location)
        end_waypoint = self.map.get_waypoint(end_location)
        
        # 使用A*算法规划路径
        route = self._astar_search(start_waypoint, end_waypoint)
        
        # 路径平滑
        smoothed_route = self._smooth_path(route)
        
        return smoothed_route
        
    def get_next_waypoints(self, current_location: carla.Location, 
                          route: List[carla.Location], 
                          look_ahead: int = 10) -> List[carla.Location]:
        """获取前方路点"""
        # 找到最近的路点索引
        closest_idx = self._get_closest_point_index(current_location, route)
        
        # 获取前方路点
        next_waypoints = route[closest_idx:closest_idx + look_ahead]
        
        return next_waypoints
        
    def _astar_search(self, start: carla.Waypoint, 
                     end: carla.Waypoint) -> List[carla.Location]:
        """A*搜索算法"""
        # 初始化开放和关闭列表
        open_list = [(0, start)]
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        
        while open_list:
            current_cost, current = open_list.pop(0)
            
            if self._is_goal(current, end):
                return self._reconstruct_path(came_from, current)
                
            closed_set.add(current)
            
            # 遍历相邻节点
            for next_wp in current.next(self.sampling_resolution):
                if next_wp in closed_set:
                    continue
                    
                tentative_g = g_score[current] + self.sampling_resolution
                
                if next_wp not in g_score or tentative_g < g_score[next_wp]:
                    came_from[next_wp] = current
                    g_score[next_wp] = tentative_g
                    f_score = tentative_g + self._heuristic(next_wp, end)
                    open_list.append((f_score, next_wp))
                    open_list.sort(key=lambda x: x[0])
                    
        return []
        
    def _smooth_path(self, route: List[carla.Location]) -> List[carla.Location]:
        """路径平滑"""
        if len(route) <= 2:
            return route
            
        # 使用三次样条插值
        points = np.array([[p.x, p.y, p.z] for p in route])
        t = np.linspace(0, 1, len(points))
        t_new = np.linspace(0, 1, len(points) * 5)
        
        smoothed_points = []
        for i in range(3):  # x, y, z
            coeffs = np.polyfit(t, points[:, i], 3)
            smoothed_points.append(np.polyval(coeffs, t_new))
            
        # 转换回carla.Location
        smoothed_route = []
        for x, y, z in zip(*smoothed_points):
            smoothed_route.append(carla.Location(x=x, y=y, z=z))
            
        return smoothed_route 