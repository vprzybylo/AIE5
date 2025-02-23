import pytest
from src.rag.document_loader import GridCodeLoader
from src.embedding.model import EmbeddingModel
from src.rag.vectorstore import VectorStore

def test_document_loading():
    loader = GridCodeLoader("tests/data/test_grid_code.txt")
    docs = loader.load_and_split()
    assert len(docs) > 0

def test_embedding_model():
    model = EmbeddingModel()
    embeddings = model.get_embeddings(["test text"])
    assert len(embeddings) == 1

def test_vectorstore():
    model = EmbeddingModel()
    store = VectorStore(model)
    # Add more specific tests 