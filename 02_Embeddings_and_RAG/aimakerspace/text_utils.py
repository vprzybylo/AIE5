import os
from typing import List, Dict
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from dataclasses import dataclass
from rich import print as rprint

@dataclass
class Document:
    page_content: str
    metadata: Dict

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8", topic: str = "LLM"):
        self.documents = []
        self.path = path
        self.encoding = encoding
        self.topic = topic
        self.date_added = datetime.now().isoformat()

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_pdf_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt or .pdf file."
            )

    def load_file(self):
        filename = os.path.basename(self.path)

        with open(self.path, "r", encoding=self.encoding) as f:
            content = f.read()
            metadata = {
                "filename": filename,
                "date_added": self.date_added,
                "topic": self.topic,
                "source": self.path
            }
            self.documents.append(Document(page_content=content, metadata=metadata))

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    full_path = os.path.join(root, file)
                    with open(full_path, "r", encoding=self.encoding) as f:
                        content = f.read()
                        metadata = {
                            "filename": file,
                            "date_added": self.date_added,
                            "topic": self.topic,
                            "source": full_path
                        }
                        self.documents.append(Document(page_content=content, metadata=metadata))

    def load_pdf_file(self):
        pdf_loader = PyPDFLoader(self.path)
        pdf_documents = pdf_loader.load()
        
        for pdf_doc in pdf_documents:
            metadata = {
                "filename": os.path.basename(self.path),
                "date_added": self.date_added,
                "topic": self.topic,
                "source": self.path,
                "page": pdf_doc.metadata['page']
            }
            self.documents.append(Document(page_content=pdf_doc.page_content, metadata=metadata))

    def load_documents(self) -> List[Document]:
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 3000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: Dict = None) -> List[Document]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i : i + self.chunk_size]
            chunks.append(Document(page_content=chunk_text, metadata=metadata.copy() if metadata else {}))
        return chunks

    def split_texts(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            chunks.extend(self.split(doc.page_content, doc.metadata))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    documents = loader.load_documents()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(documents)
    print(len(chunks))
    print(f"Content: {chunks[0].page_content[:100]}")
    print(f"Metadata: {chunks[0].metadata}")
    print("--------")
    print(f"Content: {chunks[-1].page_content[-100:]}")
    print(f"Metadata: {chunks[-1].metadata}")
