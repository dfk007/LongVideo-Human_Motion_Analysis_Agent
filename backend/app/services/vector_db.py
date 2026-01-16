import chromadb
from chromadb.config import Settings
from app.core.config import settings
from typing import List, Dict
import time
import logging

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        self._connect_with_retry()

    def _connect_with_retry(self, max_retries=10, delay=3):
        host = settings.CHROMA_HOST
        port = settings.CHROMA_PORT
        logger.info(f"Connecting to ChromaDB at {host}:{port}...")
        
        for attempt in range(max_retries):
            try:
                # Connect to ChromaDB container
                self.client = chromadb.HttpClient(
                    host=host,
                    port=port
                )
                
                # Test connection
                self.client.heartbeat()
                
                # Get or create collection
                self.collection = self.client.get_or_create_collection(
                    name="video_segments"
                )
                logger.info("Successfully connected to ChromaDB")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to ChromaDB (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    logger.error("Could not connect to ChromaDB after multiple attempts")
                    raise e

    def add_segments(self, segments_data: List[Dict]):
        """
        segments_data expected format:
        [
            {
                "id": "video1_001",
                "text": "Athlete performing a squat...",
                "metadata": {"start": 0, "end": 15, "source": "video1"}
            }
        ]
        """
        if not segments_data:
            return

        ids = [item["id"] for item in segments_data]
        documents = [item["text"] for item in segments_data]
        metadatas = [item["metadata"] for item in segments_data]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search_similar(self, query: str, n_results: int = 5):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

vector_db = VectorDBService()
