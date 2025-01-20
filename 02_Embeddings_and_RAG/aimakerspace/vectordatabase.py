import numpy as np
from typing import List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.text_utils import Document
import asyncio


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """
    Calculate the Manhattan distance between two points.

    :param point1: A list or tuple representing the first point.
    :param point2: A list or tuple representing the second point.
    :return: The Manhattan distance between the two points.

    Pros:
    - Robustness to Outliers: Less sensitive to outliers compared to Euclidean distance, as it does not square the differences.
    - Interpretability: The distance can be easily interpreted in terms of actual movement in a grid-like environment.
    - Simplicity: The calculation is straightforward and computationally efficient.

    Cons:
    - Sensitivity to Scale: If the features have different scales, the Manhattan distance can be affected; normalization may be necessary.
    - Less Sensitive to Direction: Unlike cosine similarity, Manhattan distance does not take into account the direction of the vectors, which can be a limitation in certain applications.
    - Limited Applicability: It may not perform well in high-dimensional spaces where the geometry of the data is more complex.

    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Points must have the same dimension.")
    
    return sum(abs(a - b) for a, b in zip(vector_a, vector_b))



class VectorDatabase:
    def __init__(self):
        self.vectors = {}  
        self.documents = {} 
        self.embedding_model = EmbeddingModel()

    def insert(self, document: Document, vector: np.array) -> None:
        """Insert a document and its vector into the database"""
        doc_id = id(document)  
        self.vectors[doc_id] = vector
        self.documents[doc_id] = document

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents using vector similarity"""
        scores = [
            (self.documents[doc_id], distance_measure(query_vector, vector))
            for doc_id, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents using text query"""
        query_vector = self.embedding_model.get_embedding(query_text)
        return self.search(query_vector, k, distance_measure)

    async def abuild_from_list(self, documents: List[Document]) -> "VectorDatabase":
        """Build database from a list of Documents"""
        # Get embeddings for document content
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedding_model.async_get_embeddings(texts)
        
        # Insert documents and their embeddings
        for doc, embedding in zip(documents, embeddings):
            self.insert(doc, np.array(embedding))
        
        return self


if __name__ == "__main__":
    # Example usage with Document class
    documents = [
        Document(
            page_content="I like to eat broccoli and bananas.",
            metadata={"category": "food"}
        ),
        Document(
            page_content="I ate a banana and spinach smoothie for breakfast.",
            metadata={"category": "food"}
        ),
        Document(
            page_content="Chinchillas and kittens are cute.",
            metadata={"category": "pets"}
        ),
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(documents))
    
    results = vector_db.search_by_text("What foods are mentioned?", k=2)
    for doc, score in results:
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print(f"Score: {score}\n")
