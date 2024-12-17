import carla
import numpy as np
from typing import Dict, List, Optional
import cv2
import os
from datetime import datetime

class ReplaySystem:
    """场景回放系统"""
    def __init__(self, client: carla.Client, config: Dict):
        self.client = client
        self.config = config
        self.recording = False
        self.current_recording = None
        
        # 回放配置
        self.save_dir = config.get('save_dir', 'experiments/replays')
        self.camera_width = config.get('camera_width', 1280)
        self.camera_height = config.get('camera_height', 720)
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
    def start_recording(self):
        """开始记录"""
        if not self.recording:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_recording = os.path.join(self.save_dir, f'recording_{timestamp}.log')
            self.client.start_recorder(self.current_recording)
            self.recording = True
            
    def stop_recording(self):
        """停止记录"""
        if self.recording:
            self.client.stop_recorder()
            self.recording = False
            
    def replay(self, recording_path: Optional[str] = None, camera_follow: bool = True):
        """回放场景"""
        # 使用最新的记录或指定的记录
        recording_path = recording_path or self.current_recording
        if not recording_path or not os.path.exists(recording_path):
            raise ValueError("No recording file specified or found")
            
        # 设置回放相机
        if camera_follow:
            spectator = self.client.get_world().get_spectator()
            # 获取ego vehicle
            vehicles = self.client.get_world().get_actors().filter('vehicle.*')
            if vehicles:
                ego_vehicle = vehicles[0]  # 假设第一个车辆是ego vehicle
                # 设置相机跟随
                camera_transform = carla.Transform(
                    carla.Location(x=-8, z=6),
                    carla.Rotation(pitch=15)
                )
                spectator.set_transform(
                    ego_vehicle.get_transform() * camera_transform
                )
                
        # 开始回放
        self.client.replay_file(recording_path, start=0, duration=0, camera=0)
        
    def save_replay_frame(self, frame: np.ndarray, frame_number: int):
        """保存回放帧"""
        if self.current_recording:
            frame_dir = self.current_recording.replace('.log', '_frames')
            os.makedirs(frame_dir, exist_ok=True)
            
            frame_path = os.path.join(frame_dir, f'frame_{frame_number:06d}.jpg')
            cv2.imwrite(frame_path, frame)
            
    def get_replay_info(self, recording_path: Optional[str] = None) -> Dict:
        """获取回放信息"""
        recording_path = recording_path or self.current_recording
        if not recording_path:
            return {}
            
        info = self.client.show_recorder_file_info(recording_path)
        return {
            'path': recording_path,
            'duration': info.duration,
            'map': info.map,
            'date': info.date,
            'frames': info.frames
        } 