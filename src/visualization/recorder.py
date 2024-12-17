"""视频记录器"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict
import time
import json

class VideoRecorder:
    """视频记录器"""
    def __init__(self, config: Dict):
        self.config = config
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 视频写入器
        self.video_writer = None
        self.current_episode = 0
        
        # 指标记录
        self.metrics_history = []
        
    def record_frame(self, frame: np.ndarray, info: Dict):
        """记录帧"""
        if self.config['video']['enabled']:
            # 确保帧格式正确
            if frame.dtype == np.float32:
                frame = (frame * 255).astype(np.uint8)
            if frame.shape[0] == 3:  # CHW -> HWC
                frame = np.transpose(frame, (1, 2, 0))
                
            # 创建或获取视频写入器
            if self.video_writer is None:
                self._create_video_writer(frame.shape[1], frame.shape[0])
                
            # 写入帧
            self.video_writer.write(frame)
            
        # 记录指标
        if self.config['metrics']['enabled']:
            self.metrics_history.append({
                'timestamp': time.time(),
                **info
            })
            
    def end_episode(self):
        """结束回合"""
        # 关闭视频写入器
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
        # 保存指标
        if self.metrics_history:
            metrics_path = self.save_dir / f"episode_{self.current_episode:05d}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=4)
            self.metrics_history = []
            
        self.current_episode += 1
        
    def _create_video_writer(self, width: int, height: int):
        """创建视频写入器"""
        video_path = self.save_dir / f"episode_{self.current_episode:05d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.config['video']['fps'],
            (width, height)
        ) 