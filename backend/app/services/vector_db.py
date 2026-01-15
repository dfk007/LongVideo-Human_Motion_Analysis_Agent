import chromadb
from chromadb.config import Settings
from app.core.config import settings
from typing import List, Dict

class VectorDBService:
    def __init__(self):
        # Connect to ChromaDB container
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="video_segments"
            # embedding_function defaults to all-MiniLM-L6-v2
        )

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
