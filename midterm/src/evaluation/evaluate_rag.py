import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

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


from embedding.model import EmbeddingModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag.chain import RAGChain
from rag.document_loader import GridCodeLoader
from rag.vectorstore import VectorStore
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.testset import TestsetGenerator


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
        llm=generator_llm, embedding_model=generator_embeddings
    )

    # Generate synthetic test dataset
    dataset = generator.generate_with_langchain_docs(
        documents, testset_size=n_questions
    )

    logger.info(f"Generated synthetic dataset with {len(dataset)} test cases")
    return dataset.to_pandas()


@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def get_rag_response(rag_chain, question):
    """Get RAG response with retry logic"""
    return rag_chain.invoke(question)


def evaluate_rag_system(rag_chain, test_dataset):
    """Evaluate RAG system using RAGAS metrics"""
    logger.info("Starting RAGAS evaluation...")

    # Get RAG responses for each question
    eval_data = []

    # Iterate through DataFrame rows
    for _, row in test_dataset.iterrows():
        # Add delay between requests
        time.sleep(3)  # Wait 3 seconds between requests
        response = get_rag_response(rag_chain, row["user_input"])
        eval_data.append(
            {
                "user_input": row["user_input"],
                "response": response["answer"],
                "retrieved_contexts": [doc.page_content for doc in response["context"]],
                "ground_truth": row["reference"],  # Keep for faithfulness
                "reference": row["reference"],  # Keep for context_recall
            }
        )
        logger.info(f"Processed question: {row['user_input'][:50]}...")

    # Convert to pandas then to EvaluationDataset
    eval_df = pd.DataFrame(eval_data)
    logger.info("Sample evaluation data:")
    logger.info(eval_df.iloc[0].to_dict())
    eval_dataset = EvaluationDataset.from_pandas(eval_df)

    # Initialize RAGAS evaluator
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

    custom_run_config = RunConfig(timeout=360, max_workers=32)

    # Run evaluation
    results = evaluate(
        eval_dataset,
        metrics=[
            Faithfulness(),  # Measures how accurately the generated response reflects the ground truth.
            AnswerRelevancy(),  # Assesses the relevance of the answer to the user's question.
            ContextRecall(),  # Evaluates the ability of the model to retrieve relevant context from the documents.
            ContextPrecision(),  # Measures the precision of the retrieved contexts in relation to the user's question.
        ],
        llm=evaluator_llm,
        run_config=custom_run_config,
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
        logger.info(results)

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
