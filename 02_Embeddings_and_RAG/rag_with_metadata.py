import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from datetime import datetime

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Load PDF
pdf_path = "data/test.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Add metadata and modify document content
for i, split in enumerate(splits):
    access_date = datetime.now().isoformat()

    # Document identification
    split.metadata["source"] = f"page_{split.metadata['page']}"
    split.metadata["doc_id"] = f"doc_{os.path.basename(pdf_path)}_{i}"
    split.metadata["chunk_index"] = i

    # Content classification
    split.metadata["topic"] = "LLM"
    split.metadata["content_type"] = (
        "research_paper" 
    )
    split.metadata["language"] = "en"
    split.metadata["processing_date"] = access_date
    split.metadata["total_chunks"] = len(splits)

    # Add key metadata to content for embedding
    split.page_content = f"""
        Document ID: {split.metadata["doc_id"]}
        Content Type: {split.metadata["content_type"]}
        Processing Date: {access_date}

        {split.page_content}
    """

embedding_function = OpenAIEmbeddings()

# Set up vector database using FAISS
# Note I couldn't get the embedded metadata working with the vectorstore in aimakerspace
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_function)
retriever = vectorstore.as_retriever()

# Set up language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


# Function to ask questions
def ask_question(question):
    result = qa_chain.invoke({"query": question})
    answer = result["result"]

    # Collect relevant metadata
    doc_ids = [doc.metadata["doc_id"] for doc in result["source_documents"]]
    content_types = [doc.metadata["content_type"] for doc in result["source_documents"]]

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Document IDs: {', '.join(set(doc_ids))}")
    print(f"Content Types: {', '.join(set(content_types))}")
    print("\n")


# Example usage
ask_question("What is the main topic of the PDF?")
ask_question("When was this document processed or accessed by the system and what is the content type?")
