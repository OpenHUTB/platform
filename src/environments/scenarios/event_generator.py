import carla
import numpy as np
from typing import Dict, List, Tuple
import random

class EventGenerator:
    """场景事件生成器"""
    def __init__(self, world: carla.World):
        self.world = world
        self.event_handlers = {
            'sudden_braking': self._handle_sudden_braking,
            'pedestrian_crossing': self._handle_pedestrian_crossing,
            'vehicle_cutting_in': self._handle_vehicle_cutting_in,
            'traffic_light_change': self._handle_traffic_light_change
        }
        
    def generate_event(self, event_type: str, params: Dict = None):
        """生成事件"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type](params or {})
            
    def _handle_sudden_braking(self, params: Dict):
        """处理突然刹车事件"""
        # 获取前方车辆
        vehicle = self._get_front_vehicle()
        if vehicle:
            # 设置刹车控制
            control = vehicle.get_control()
            control.brake = 1.0
            vehicle.apply_control(control)
            
    def _handle_pedestrian_crossing(self, params: Dict):
        """处理行人横穿事件"""
        # 获取附近行人
        walker = self._get_nearby_walker()
        if walker:
            # 设置横穿路径
            crossing_path = self._generate_crossing_path(walker)
            self._set_walker_path(walker, crossing_path)
            
    def _handle_vehicle_cutting_in(self, params: Dict):
        """处理车辆切入事件"""
        # 获取旁边车道的车辆
        vehicle = self._get_adjacent_vehicle()
        if vehicle:
            # 设置切入轨迹
            cutting_path = self._generate_cutting_path(vehicle)
            self._set_vehicle_path(vehicle, cutting_path)
            
    def _handle_traffic_light_change(self, params: Dict):
        """处理信号灯变化事件"""
        # 获取前方信号灯
        traffic_light = self._get_front_traffic_light()
        if traffic_light:
            # 改变信号灯状态
            self._change_traffic_light_state(traffic_light)
            
    def _get_front_vehicle(self, distance: float = 50.0) -> carla.Vehicle:
        """获取前方车辆"""
        # 获取自车位置和朝向
        ego_location = self.ego_vehicle.get_location()
        ego_transform = self.ego_vehicle.get_transform()
        ego_forward = ego_transform.get_forward_vector()
        
        # 获取所有车辆
        vehicles = self.world.get_actors().filter('vehicle.*')
        
        # 找到前方最近的车辆
        min_distance = float('inf')
        front_vehicle = None
        
        for vehicle in vehicles:
            if vehicle.id == self.ego_vehicle.id:
                continue
            
            # 计算相对位置
            relative_pos = vehicle.get_location() - ego_location
            
            # 计算前向距离
            forward_distance = relative_pos.x * ego_forward.x + relative_pos.y * ego_forward.y
            
            # 检查是否在前方
            if forward_distance > 0 and forward_distance < distance:
                # 计算横向距离
                lateral_distance = abs(relative_pos.x * ego_forward.y - relative_pos.y * ego_forward.x)
                
                # 检查是否在同一车道
                if lateral_distance < 2.0 and forward_distance < min_distance:
                    min_distance = forward_distance
                    front_vehicle = vehicle
                    
        return front_vehicle

    def _get_nearby_walker(self, radius: float = 20.0) -> carla.Walker:
        """获取附近行人"""
        # 获取自车位置
        ego_location = self.ego_vehicle.get_location()
        
        # 获取所有行人
        walkers = self.world.get_actors().filter('walker.*')
        
        # 找到最近的行人
        min_distance = float('inf')
        nearest_walker = None
        
        for walker in walkers:
            # 计算距离
            distance = walker.get_location().distance(ego_location)
            
            if distance < radius and distance < min_distance:
                min_distance = distance
                nearest_walker = walker
                
        return nearest_walker

    def _get_adjacent_vehicle(self, distance: float = 30.0) -> carla.Vehicle:
        """获取相邻车道车辆"""
        # 获取自车位置和车道
        ego_location = self.ego_vehicle.get_location()
        ego_waypoint = self.world.get_map().get_waypoint(ego_location)
        
        # 获取相邻车道
        adjacent_lanes = []
        if ego_waypoint.get_left_lane():
            adjacent_lanes.append(ego_waypoint.get_left_lane())
        if ego_waypoint.get_right_lane():
            adjacent_lanes.append(ego_waypoint.get_right_lane())
        
        # 获取所有车辆
        vehicles = self.world.get_actors().filter('vehicle.*')
        
        # 找到相邻车道最近的车辆
        min_distance = float('inf')
        adjacent_vehicle = None
        
        for vehicle in vehicles:
            if vehicle.id == self.ego_vehicle.id:
                continue
            
            # 获取车辆车道
            vehicle_waypoint = self.world.get_map().get_waypoint(vehicle.get_location())
            
            # 检查是否在相邻车道
            if vehicle_waypoint in adjacent_lanes:
                # 计算距离
                distance = vehicle.get_location().distance(ego_location)
                
                if distance < min_distance:
                    min_distance = distance
                    adjacent_vehicle = vehicle
                    
        return adjacent_vehicle

    def _generate_crossing_path(self, walker: carla.Walker) -> List[carla.Location]:
        """生成行人横穿路径"""
        # 获取当前位置
        current_loc = walker.get_location()
        
        # 获取最近的人行道
        waypoint = self.world.get_map().get_waypoint(current_loc)
        sidewalk = waypoint.get_right_lane()
        
        # 生成路径点
        path = []
        path.append(current_loc)
        
        # 添加中间点
        mid_point = carla.Location(
            x=(current_loc.x + sidewalk.transform.location.x) / 2,
            y=(current_loc.y + sidewalk.transform.location.y) / 2,
            z=current_loc.z
        )
        path.append(mid_point)
        
        # 添加终点
        path.append(sidewalk.transform.location)
        
        return path

    def _generate_cutting_path(self, vehicle: carla.Vehicle) -> List[carla.Location]:
        """生成车辆切入轨迹"""
        # 获取当前位置和车道
        current_wp = self.world.get_map().get_waypoint(vehicle.get_location())
        
        # 获取目标车道
        target_wp = current_wp.get_left_lane() or current_wp.get_right_lane()
        if not target_wp:
            return []
        
        # 生成轨迹点
        path = []
        path.append(current_wp.transform.location)
        
        # 添加中间点
        for i in range(5):
            alpha = (i + 1) / 6
            mid_point = carla.Location(
                x=(1-alpha)*current_wp.transform.location.x + alpha*target_wp.transform.location.x,
                y=(1-alpha)*current_wp.transform.location.y + alpha*target_wp.transform.location.y,
                z=current_wp.transform.location.z
            )
            path.append(mid_point)
        
        # 添加终点
        path.append(target_wp.transform.location)
        
        return path

    def _set_walker_path(self, walker: carla.Walker, path: List[carla.Location]):
        """设置行人路径"""
        # 创建行人控制器
        controller = self.world.spawn_actor(
            self.world.get_blueprint_library().find('controller.ai.walker'),
            carla.Transform(),
            walker
        )
        
        # 设置路径点
        controller.start()
        controller.go_to_location(path[-1])
        
        # 设置行走速度
        controller.set_max_speed(1.4)  # 正常行走速度
        
    def _set_vehicle_path(self, vehicle: carla.Vehicle, path: List[carla.Location]):
        """设置车辆路径"""
        # 创建路径规划器
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1.0)
        
        # 生成详细路径
        route = []
        for i in range(len(path)-1):
            # 获取当前点和下一点的路径
            current_route = grp.trace_route(path[i], path[i+1])
            route.extend(current_route)
        
        # 设置车辆行为
        behavior = self._get_vehicle_behavior(vehicle)
        behavior.set_route(route)
        
    def _get_vehicle_behavior(self, vehicle: carla.Vehicle) -> VehicleBehavior:
        """获取车辆行为控制器"""
        # 创建行为控制器
        behavior = VehicleBehavior(vehicle)
        
        # 设置基本参数
        behavior.target_speed = 30.0  # km/h
        behavior.safety_distance = 5.0  # meters
        behavior.max_steering = 0.7
        behavior.max_brake = 0.3
        
        return behavior

class VehicleBehavior:
    """车辆行为控制器"""
    def __init__(self, vehicle: carla.Vehicle):
        self.vehicle = vehicle
        self.route = []
        self.current_waypoint_index = 0
        
        # 控制参数
        self.target_speed = 30.0
        self.safety_distance = 5.0
        self.max_steering = 0.7
        self.max_brake = 0.3
        
    def set_route(self, route: List[Tuple[carla.Waypoint, str]]):
        """设置路径"""
        self.route = route
        self.current_waypoint_index = 0
        
    def update(self):
        """更新控制"""
        if not self.route:
            return
            
        # 获取当前位置和目标点
        current_transform = self.vehicle.get_transform()
        target_waypoint = self.route[self.current_waypoint_index][0]
        
        # 计算控制量
        control = self._compute_control(current_transform, target_waypoint)
        
        # 应用控制
        self.vehicle.apply_control(control)
        
        # 更新目标点
        if self._reached_waypoint(current_transform, target_waypoint):
            self.current_waypoint_index += 1
            
    def _compute_control(self, current: carla.Transform, 
                        target: carla.Waypoint) -> carla.VehicleControl:
        """计算控制量"""
        # 计算转向角
        target_vector = target.transform.location - current.location
        target_yaw = np.arctan2(target_vector.y, target_vector.x)
        current_yaw = np.radians(current.rotation.yaw)
        
        steering = self._normalize_angle(target_yaw - current_yaw)
        steering = np.clip(steering / np.pi, -self.max_steering, self.max_steering)
        
        # 计算速度控制
        current_speed = self._get_speed()
        if current_speed < self.target_speed:
            throttle = 1.0
            brake = 0.0
        else:
            throttle = 0.0
            brake = min((current_speed - self.target_speed) / 10.0, self.max_brake)
            
        return carla.VehicleControl(throttle=throttle, steer=steering, brake=brake)