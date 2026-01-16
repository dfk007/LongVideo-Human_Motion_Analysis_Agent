"""
Video Processing Workflow

This module orchestrates the complete video processing pipeline:
1. Split video into chunks
2. Extract pose data from each chunk
3. Generate descriptions using Gemini
4. Store everything in vector database

This implements the "perception" layer - separating what we see
from how we reason about it.
"""

import logging
from pathlib import Path
import json

from app.services.video_service import VideoService
from app.services.gemini_service import gemini_service
from app.services.vector_db import vector_db
from app.services.pose_estimation import PoseEstimator
from app.core.config import settings

logger = logging.getLogger(__name__)


class ProcessingWorkflow:
    """
    Orchestrates the video processing pipeline.
    
    This workflow implements efficient long-video handling by:
    - Chunking videos into manageable segments
    - Extracting pose data (perception)
    - Generating semantic descriptions
    - Indexing for fast retrieval
    """
    
    def __init__(self):
        self.video_service = VideoService()
        self.pose_estimator = PoseEstimator(
            model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
    
    async def process_video(self, video_path: str, filename: str):
        """
        Main processing pipeline for uploaded videos.
        
        This method implements the key insight from the project requirements:
        "Handle long videos efficiently (not brute force)" by:
        1. Segmenting into chunks (not processing entire video at once)
        2. Extracting structured data (pose) rather than just pixels
        3. Creating searchable index for targeted retrieval
        
        Args:
            video_path: Path to uploaded video file
            filename: Original filename
        """
        logger.info(f"Starting processing for {filename}")
        
        # Step 1: Split Video into Chunks
        # This enables efficient processing of long videos
        try:
            chunks = self.video_service.split_video(
                video_path, 
                chunk_duration=settings.CHUNK_DURATION
            )
            logger.info(f"Split into {len(chunks)} chunks of {settings.CHUNK_DURATION}s each")
        except Exception as e:
            logger.error(f"Failed to split video: {e}")
            return
        
        # Step 2: Process Each Chunk
        segments_to_index = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk['chunk_id']}")
            
            try:
                # Step 2a: Extract Pose Data (Perception Layer)
                # This converts raw video into interpretable measurements
                pose_data = self._extract_pose_from_chunk(chunk)
                
                # Step 2b: Generate Semantic Description (Using LLM)
                # This creates searchable text representation
                description = self._generate_chunk_description(chunk, pose_data)
                
                # Step 2c: Save pose data for later retrieval
                self._save_pose_data(chunk, pose_data)
                
                # Step 2d: Prepare for vector database
                segments_to_index.append({
                    "id": chunk['chunk_id'],
                    "text": description,
                    "metadata": {
                        "start_time": str(chunk['start_time']),
                        "end_time": str(chunk['end_time']),
                        "video_source": chunk['video_source'],
                        "chunk_path": chunk['path'],
                        "has_pose_data": len(pose_data) > 0,
                        "pose_confidence": self._calculate_avg_confidence(pose_data)
                    }
                })
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk['chunk_id']}: {e}")
                # Continue with next chunk even if one fails
                continue
        
        # Step 3: Index in Vector Database
        # This enables semantic search: "find squat moments" without exact keywords
        try:
            vector_db.add_segments(segments_to_index)
            logger.info(f"Successfully indexed {len(segments_to_index)} segments")
        except Exception as e:
            logger.error(f"Failed to index segments: {e}")
        
        logger.info(f"Completed processing for {filename}")
    
    def _extract_pose_from_chunk(self, chunk: dict) -> list:
        """
        Extract pose data from a video chunk.
        
        This implements the "perception" layer:
        - Uses MediaPipe to detect human pose
        - Calculates joint angles
        - Returns structured, interpretable data
        
        This is SEPARATE from reasoning - we just observe what's there,
        we don't judge if it's good or bad yet.
        """
        try:
            pose_sequence = self.pose_estimator.extract_pose_from_video(
                chunk['path'],
                sample_rate=settings.FRAME_SAMPLE_RATE
            )
            
            logger.info(
                f"Extracted {len(pose_sequence)} poses from chunk {chunk['chunk_id']}"
            )
            
            return pose_sequence
            
        except Exception as e:
            logger.error(f"Error extracting pose from chunk {chunk['chunk_id']}: {e}")
            return []
    
    def _generate_chunk_description(self, chunk: dict, pose_data: list) -> str:
        """
        Generate a semantic description of the chunk using Gemini.
        
        This creates a searchable text representation that includes:
        - What movement is happening
        - Key biomechanical observations
        - Approximate measurements
        
        The description is what gets embedded and searched later.
        """
        try:
            # If we have pose data, include it in the prompt for better description
            pose_summary = ""
            if pose_data:
                # Calculate statistics from pose data
                analyzer = self.pose_estimator.analyze_motion_sequence(pose_data)
                
                pose_summary = "\n\nPose Data Summary:\n"
                for angle_name, stats in analyzer.get('angle_ranges', {}).items():
                    pose_summary += (
                        f"- {angle_name}: {stats['min']:.1f}° to {stats['max']:.1f}° "
                        f"(mean: {stats['mean']:.1f}°)\n"
                    )
            
            # Generate description using Gemini
            description = gemini_service.describe_segment(
                chunk['path']
            )
            
            # Append pose data summary to description
            if pose_summary:
                description = description + pose_summary
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating description for chunk {chunk['chunk_id']}: {e}")
            return f"Video segment from {chunk['start_time']}s to {chunk['end_time']}s"
    
    def _save_pose_data(self, chunk: dict, pose_data: list):
        """
        Save pose data to disk for later retrieval.
        
        We save this separately so we can load it quickly when the agent
        needs detailed measurements without re-processing the video.
        """
        if not pose_data:
            return
        
        try:
            # Create pose data directory
            pose_data_dir = Path(settings.PROCESSED_DIR) / chunk['video_source'] / 'pose_data'
            pose_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            output_path = pose_data_dir / f"{chunk['chunk_id']}_pose.json"
            self.pose_estimator.save_pose_data(pose_data, str(output_path))
            
            logger.info(f"Saved pose data to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving pose data for chunk {chunk['chunk_id']}: {e}")
    
    def _calculate_avg_confidence(self, pose_data: list) -> float:
        """Calculate average detection confidence across all poses"""
        if not pose_data:
            return 0.0
        
        confidences = [p.confidence for p in pose_data]
        return float(sum(confidences) / len(confidences))


# Global instance
workflow = ProcessingWorkflow()
