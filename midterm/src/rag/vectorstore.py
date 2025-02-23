from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.collection_name = "grid_code"
        logger.info("Initialized VectorStore in memory mode")
    
    def create_vectorstore(self, documents):
        # Initialize Qdrant client in memory mode
        client = QdrantClient(":memory:")

        try:
            # Try to get the collection info
            client.get_collection(self.collection_name)
            logger.info("Found existing collection, reusing vectors")
            return Qdrant(
                client=client,
                collection_name=self.collection_name,
                embeddings=self.embedding_model.model
            )
        except Exception as e:
            logger.info("No existing collection found, creating new vectors")
            # Create new collection and index documents
            return Qdrant.from_documents(
                documents=documents,
                embedding=self.embedding_model.model,
                location=":memory:",
                collection_name=self.collection_name
            )
    
    def similarity_search(self, query, k=4):
        raise NotImplementedError("Use the Qdrant vectorstore instance directly") 