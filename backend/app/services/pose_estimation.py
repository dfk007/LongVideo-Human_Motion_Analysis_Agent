"""
Pose Estimation Service using MediaPipe

This service extracts human pose landmarks from video frames and calculates
biomechanical metrics like joint angles, positions, and movement patterns.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


@dataclass
class PoseData:
    """Container for pose data from a single frame"""
    frame_number: int
    timestamp: float
    landmarks: Dict[str, Dict[str, float]]  # landmark_name -> {x, y, z, visibility}
    angles: Dict[str, float]  # joint_name -> angle in degrees
    confidence: float
    

class PoseEstimator:
    """
    Extracts and analyzes human pose from video frames using MediaPipe.
    
    This class handles:
    - Pose landmark detection
    - Joint angle calculation
    - Motion metrics extraction
    - Safety analysis preparation
    """
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the pose estimator.
        
        Args:
            model_complexity: 0=Lite, 1=Full, 2=Heavy (higher is more accurate but slower)
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Landmark indices for common joints (MediaPipe pose landmarks)
        self.landmark_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
        }
        
    def extract_pose_from_video(
        self, 
        video_path: str, 
        sample_rate: int = 3
    ) -> List[PoseData]:
        """
        Extract pose data from entire video.
        
        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (3 = process every 3rd frame)
            
        Returns:
            List of PoseData objects, one per processed frame
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        pose_data_list = []
        
        logger.info(f"Processing video: {video_path} at {fps} FPS, sampling every {sample_rate} frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process every Nth frame
            if frame_count % sample_rate == 0:
                timestamp = frame_count / fps
                pose_data = self._process_frame(frame, frame_count, timestamp)
                
                if pose_data:  # Only add if pose was detected
                    pose_data_list.append(pose_data)
                    
            frame_count += 1
            
        cap.release()
        logger.info(f"Extracted {len(pose_data_list)} pose frames from {frame_count} total frames")
        
        return pose_data_list
    
    def _process_frame(
        self, 
        frame: np.ndarray, 
        frame_number: int, 
        timestamp: float
    ) -> Optional[PoseData]:
        """
        Process a single frame to extract pose data.
        
        Args:
            frame: OpenCV frame (BGR)
            frame_number: Frame index
            timestamp: Time in seconds
            
        Returns:
            PoseData object or None if no pose detected
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
            
        # Extract landmarks
        landmarks = self._extract_landmarks(results.pose_landmarks)
        
        # Calculate angles
        angles = self._calculate_angles(landmarks)
        
        # Calculate average confidence
        confidence = float(np.mean([
            lm['visibility'] 
            for lm in landmarks.values()
        ]))
        
        return PoseData(
            frame_number=frame_number,
            timestamp=timestamp,
            landmarks=landmarks,
            angles=angles,
            confidence=confidence
        )
    
    def _extract_landmarks(self, pose_landmarks) -> Dict[str, Dict[str, float]]:
        """
        Extract landmark coordinates from MediaPipe results.
        
        Returns dict mapping landmark names to {x, y, z, visibility}
        """
        landmarks = {}
        
        for name, idx in self.landmark_indices.items():
            lm = pose_landmarks.landmark[idx]
            landmarks[name] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            }
            
        return landmarks
    
    def _calculate_angles(self, landmarks: Dict) -> Dict[str, float]:
        """
        Calculate key joint angles from landmarks.
        
        Returns dict mapping angle names to degrees
        """
        angles = {}
        
        # Left knee angle (hip -> knee -> ankle)
        if all(k in landmarks for k in ['left_hip', 'left_knee', 'left_ankle']):
            angles['left_knee'] = self._angle_between_points(
                landmarks['left_hip'],
                landmarks['left_knee'],
                landmarks['left_ankle']
            )
        
        # Right knee angle
        if all(k in landmarks for k in ['right_hip', 'right_knee', 'right_ankle']):
            angles['right_knee'] = self._angle_between_points(
                landmarks['right_hip'],
                landmarks['right_knee'],
                landmarks['right_ankle']
            )
        
        # Left elbow angle (shoulder -> elbow -> wrist)
        if all(k in landmarks for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angles['left_elbow'] = self._angle_between_points(
                landmarks['left_shoulder'],
                landmarks['left_elbow'],
                landmarks['left_wrist']
            )
        
        # Right elbow angle
        if all(k in landmarks for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angles['right_elbow'] = self._angle_between_points(
                landmarks['right_shoulder'],
                landmarks['right_elbow'],
                landmarks['right_wrist']
            )
        
        # Hip angle (shoulder -> hip -> knee) - measures forward lean
        if all(k in landmarks for k in ['left_shoulder', 'left_hip', 'left_knee']):
            angles['left_hip'] = self._angle_between_points(
                landmarks['left_shoulder'],
                landmarks['left_hip'],
                landmarks['left_knee']
            )
        
        # Back angle (approximate spinal alignment using shoulder-hip-ankle)
        if all(k in landmarks for k in ['left_shoulder', 'left_hip', 'left_ankle']):
            angles['back_angle'] = self._angle_between_points(
                landmarks['left_shoulder'],
                landmarks['left_hip'],
                landmarks['left_ankle']
            )
        
        return angles
    
    @staticmethod
    def _angle_between_points(
        p1: Dict[str, float], 
        p2: Dict[str, float], 
        p3: Dict[str, float]
    ) -> float:
        """
        Calculate angle at p2 formed by points p1-p2-p3.
        
        Args:
            p1, p2, p3: Points with 'x', 'y', 'z' coordinates
            
        Returns:
            Angle in degrees
        """
        # Create vectors
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
        
        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range for arccos
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return float(angle_deg)
    
    @staticmethod
    
    def analyze_motion_sequence(
        self, 
        pose_sequence: List[PoseData]
    ) -> Dict[str, any]:
        """
        Analyze a sequence of poses to extract motion patterns.
        
        Args:
            pose_sequence: List of PoseData objects in temporal order
            
        Returns:
            Dict with motion analysis metrics
        """
        if not pose_sequence:
            return {}
        
        analysis = {
            'duration': pose_sequence[-1].timestamp - pose_sequence[0].timestamp,
            'frame_count': len(pose_sequence),
            'avg_confidence': np.mean([p.confidence for p in pose_sequence]),
            'angle_ranges': {},
            'angle_stats': {}
        }
        
        # Analyze each angle over time
        for angle_name in pose_sequence[0].angles.keys():
            angles = [p.angles.get(angle_name, 0) for p in pose_sequence]
            
            analysis['angle_ranges'][angle_name] = {
                'min': float(np.min(angles)),
                'max': float(np.max(angles)),
                'mean': float(np.mean(angles)),
                'std': float(np.std(angles))
            }
        
        return analysis
    
    def save_pose_data(self, pose_data_list: List[PoseData], output_path: str):
        """Save pose data to JSON file"""
        data = {
            'poses': [
                {
                    'frame_number': p.frame_number,
                    'timestamp': p.timestamp,
                    'landmarks': p.landmarks,
                    'angles': p.angles,
                    'confidence': p.confidence
                }
                for p in pose_data_list
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved pose data to {output_path}")
    
    def load_pose_data(self, input_path: str) -> List[PoseData]:
        """Load pose data from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        pose_data_list = [
            PoseData(
                frame_number=p['frame_number'],
                timestamp=p['timestamp'],
                landmarks=p['landmarks'],
                angles=p['angles'],
                confidence=p['confidence']
            )
            for p in data['poses']
        ]
        
        logger.info(f"Loaded {len(pose_data_list)} poses from {input_path}")
        return pose_data_list
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()


# Convenience function for quick pose extraction
def extract_pose_from_video(video_path: str, sample_rate: int = 3) -> List[PoseData]:
    """
    Quick function to extract pose data from a video.
    
    Args:
        video_path: Path to video file
        sample_rate: Process every Nth frame
        
    Returns:
        List of PoseData objects
    """
    estimator = PoseEstimator()
    return estimator.extract_pose_from_video(video_path, sample_rate)