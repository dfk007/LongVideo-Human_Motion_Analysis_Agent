import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict
from app.core.config import settings

class VideoService:
    def __init__(self):
        pass

    async def save_upload(self, file_content: bytes, filename: str) -> str:
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        return file_path

    def split_video(self, video_path: str, chunk_duration: int = 15) -> List[Dict]:
        """
        Splits video into chunks of `chunk_duration` seconds.
        Returns a list of metadata for each chunk.
        """
        video_name = Path(video_path).stem
        output_dir = os.path.join(settings.PROCESSED_DIR, video_name)
        os.makedirs(output_dir, exist_ok=True)

        # Output pattern: video_name/chunk_001.mp4
        output_pattern = os.path.join(output_dir, "chunk_%03d.mp4")
        
        # FFmpeg command to split video
        # -c copy is fast but might be inaccurate on keyframes. 
        # Re-encoding ensure precise cuts but is slower. We'll try copy first for speed.
        # Actually, for analysis, re-encoding is safer to avoid corrupt frames at start.
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-c:a", "aac",
            "-f", "segment",
            "-segment_time", str(chunk_duration),
            "-reset_timestamps", "1",
            output_pattern
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError("Failed to process video")

        # Collect chunk info
        chunks = []
        for file in sorted(os.listdir(output_dir)):
            if file.endswith(".mp4"):
                chunk_path = os.path.join(output_dir, file)
                # Calculate simple timestamp based on index
                # Note: This is an approximation. For production, we'd parse exact start times.
                index = int(file.split('_')[1].split('.')[0])
                start_time = index * chunk_duration
                end_time = (index + 1) * chunk_duration
                
                chunks.append({
                    "chunk_id": f"{video_name}_{index}",
                    "path": chunk_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "video_source": video_name
                })
        
        return chunks
