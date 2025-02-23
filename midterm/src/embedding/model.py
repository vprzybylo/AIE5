from langchain_community.embeddings import HuggingFaceEmbedding
from langchain_openai import OpenAIEmbeddings


class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-small")
        elif self.model_type == "finetuned":
            return HuggingFaceEmbedding(model_name="../data/processed/finetuned_arctic_ft")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def embed_documents(self, texts):
        """Embed a list of texts"""
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        """Embed a single text"""
        return self.model.embed_query(text)
