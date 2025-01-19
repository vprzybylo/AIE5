import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict
from aimakerspace.openai_utils.embedding import EmbeddingModel
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
        self.vectors = defaultdict(np.array)
        self.embedding_model = EmbeddingModel()

    def insert(self, key: str, vector: np.array, metadata: Dict = None) -> None:
        self.vectors[key] = vector

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = manhattan_distance, # or manhattan distance
    ) -> List[Tuple[str, float]]:
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = manhattan_distance, # or manhattan distance
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> Tuple[np.array, Dict]:
        return self.vectors.get(key, None), self.metadata.get(key, None)


    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]


    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector, retrieved_metadata = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)
    print("Retrieved metadata:", retrieved_metadata)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
