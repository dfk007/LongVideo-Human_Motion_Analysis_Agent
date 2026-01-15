from app.services.video_service import VideoService
from app.services.gemini_service import gemini_service
from app.services.vector_db import vector_db
import logging

logger = logging.getLogger(__name__)

class ProcessingWorkflow:
    def __init__(self):
        self.video_service = VideoService()

    async def process_video(self, video_path: str, filename: str):
        logger.info(f"Starting processing for {filename}")
        
        # 1. Split Video
        try:
            # Split into 15s chunks (short enough for detailed "takeoff" analysis)
            chunks = self.video_service.split_video(video_path, chunk_duration=15)
            logger.info(f"Split into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to split video: {e}")
            return

        # 2. Analyze & Index Chunks
        segments_to_index = []
        
        for chunk in chunks:
            logger.info(f"Analyzing chunk {chunk['chunk_id']}")
            
            # Get description from Gemini
            description = gemini_service.describe_segment(chunk['path'])
            
            # Prepare data for Vector DB
            segments_to_index.append({
                "id": chunk['chunk_id'],
                "text": description,
                "metadata": {
                    "start_time": str(chunk['start_time']), # Chroma metadata must be int, float, str, or bool
                    "end_time": str(chunk['end_time']),
                    "video_source": chunk['video_source'],
                    "chunk_path": chunk['path']
                }
            })
            
        # 3. Store in Vector DB
        try:
            vector_db.add_segments(segments_to_index)
            logger.info(f"Successfully indexed {len(segments_to_index)} segments")
        except Exception as e:
            logger.error(f"Failed to index segments: {e}")

workflow = ProcessingWorkflow()
