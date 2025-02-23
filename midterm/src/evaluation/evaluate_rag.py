import sys
from pathlib import Path
import logging
from dotenv import load_dotenv
import os
import time
from tenacity import retry, wait_exponential, stop_after_attempt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

# Load environment variables
root_dir = Path(__file__).parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

from rag.document_loader import GridCodeLoader
from rag.vectorstore import VectorStore
from rag.chain import RAGChain
from embedding.model import EmbeddingModel
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision
)
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings
from rich import print


def setup_rag():
    """Initialize RAG system for evaluation"""
    logger.info("Setting up RAG system...")
    
    # Load documents
    data_path = root_dir / "data" / "raw" / "grid_code.pdf"
    if not data_path.exists():
        raise FileNotFoundError(f"PDF not found: {data_path}")
    
    loader = GridCodeLoader(str(data_path), pages=17) 
    documents = loader.load_and_split()
    logger.info(f"Loaded {len(documents)} document chunks")
    
    # Initialize embedding model and vectorstore
    embedding_model = EmbeddingModel()
    vectorstore = VectorStore(embedding_model)
    vectorstore = vectorstore.create_vectorstore(documents)
    
    return RAGChain(vectorstore), documents

def generate_test_dataset(documents, n_questions=10):
    """Generate synthetic test dataset using RAGAS"""
    logger.info("Generating synthetic test dataset...")
    
    # Initialize generator models
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    # Create test set generator
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )
    
    # Generate synthetic test dataset
    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=n_questions
    )
    
    logger.info(f"Generated synthetic dataset with {len(dataset)} test cases")
    return dataset

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def get_rag_response(rag_chain, question):
    """Get RAG response with retry logic"""
    return rag_chain.invoke(question)

def evaluate_rag_system(rag_chain, test_dataset):
    """Evaluate RAG system using RAGAS metrics"""
    logger.info("Starting RAGAS evaluation...")
    
    # Get RAG responses for each question
    eval_data = []
    for item in test_dataset:
        # Add delay between requests
        time.sleep(3)  # Wait 3 seconds between requests
        response = get_rag_response(rag_chain, item.user_input)
        eval_data.append({
            "question": item.user_input,
            "answer": response["answer"],
            "contexts": [doc.page_content for doc in response["context"]],
            "ground_truths": [item.reference]  # Using reference as ground truth
        })
        logger.info(f"Processed question: {item.user_input[:50]}...")
    
    # Initialize RAGAS evaluator
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    
    # Run evaluation
    results = evaluate(
        eval_data,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextRecall(),
            ContextPrecision()
        ],
        llm=evaluator_llm
    )
    
    return results

def main():
    """Main evaluation script"""
    logger.info("Starting RAG evaluation")
    
    try:
        # Setup RAG system
        rag_chain, documents = setup_rag()
        
        # Generate synthetic test dataset
        test_dataset = generate_test_dataset(documents)
        
        # Run evaluation
        results = evaluate_rag_system(rag_chain, test_dataset)
        
        # Print results
        logger.info("Evaluation Results:")
        for metric, score in results.items():
            logger.info(f"{metric}: {score:.3f}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 