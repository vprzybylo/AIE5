import os
from typing import List, Dict, Tuple
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader

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
            # Append content and metadata as a tuple
            self.documents.append((content, {
                "filename": filename,
                "date_added": self.date_added,
                "topic": self.topic
            }))

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    full_path = os.path.join(root, file)
                    self.load_file_with_metadata(full_path)


    def load_pdf_file(self):
        pdf_loader = PyPDFLoader(self.path)
        pdf_documents = pdf_loader.load()

        for pdf_doc in pdf_documents:
            content = pdf_doc.page_content
            metadata = {
                "filename": os.path.basename(self.path),
                "date_added": self.date_added,
                "topic": self.topic,
                "page": pdf_doc.metadata['page']
            }
            self.documents.append((content, metadata))

    def load_documents(self) -> List[Tuple[str, Dict]]:
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        print(type(text))
        for i in range(0, len(text.page_content), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
