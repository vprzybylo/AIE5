import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

# Create data directories if they don't exist
def setup_data_directories():
    root_dir = Path(__file__).parent.parent.parent
    data_dirs = [
        root_dir / "data" / "raw",
        root_dir / "data" / "processed",
        root_dir / "data" / "processed" / "qdrant"
    ]
    
    for dir_path in data_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

# Load environment variables from .env file in project root
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
logger.info(f"Loading environment from: {env_path}")
load_dotenv(env_path)

# Ensure data directories exist
setup_data_directories()

# Verify OpenAI API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OpenAI API key not found")
    st.error("OpenAI API key not found. Please ensure it is set in your .env file.")
    st.stop()
else:
    logger.info("OpenAI API key loaded successfully")

from rag.document_loader import GridCodeLoader
from rag.vectorstore import VectorStore
from rag.chain import RAGChain
from embedding.model import EmbeddingModel

def initialize_rag():
    # Get absolute path to the data file
    root_dir = Path(__file__).parent.parent.parent
    data_path = root_dir / "data" / "raw" / "grid_code.pdf"
    
    if not data_path.exists():
        logger.error(f"PDF not found: {data_path}")
        st.error(f"Grid Code PDF not found at {data_path}. Please ensure the file exists.")
        st.stop()
    
    logger.info("Loading and processing documents...")
    # Load just first 5 pages for testing
    loader = GridCodeLoader(str(data_path), pages=7)
    documents = loader.load_and_split()
    logger.info(f"Split documents into {len(documents)} chunks")
    
    logger.info("Initializing embedding model...")
    embedding_model = EmbeddingModel()
    vectorstore = VectorStore(embedding_model)
    vectorstore = vectorstore.create_vectorstore(documents)
    logger.info("Vector store created successfully")
    
    logger.info("Initializing RAG chain...")
    return RAGChain(vectorstore)

def main():
    st.title("Grid Code Assistant")
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = initialize_rag()
        logger.info("RAG chain initialized successfully")
    
    question = st.text_input("Ask a question about the Grid Code:")
    
    if question:
        logger.info(f"Processing question: {question}")
        with st.spinner("Finding answer..."):
            response = st.session_state.rag_chain.invoke(question)
            logger.info("Generated response successfully")
            st.write(response["answer"])

if __name__ == "__main__":
    main() 