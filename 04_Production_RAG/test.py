from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
import nest_asyncio

load_dotenv()

nest_asyncio.apply()

documents = SitemapLoader(web_path="https://blog.langchain.dev/sitemap-posts.xml").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=len,
)

split_chunks = text_splitter.split_documents(documents)

max_chunk_length = max(len(chunk.page_content) for chunk in split_chunks)
print(max_chunk_length)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_vectorstore = Qdrant.from_documents(
    documents=split_chunks,
    embedding=embedding_model,
    location=":memory:"
)

qdrant_retriever = qdrant_vectorstore.as_retriever()

# Define the base RAG prompt template
base_rag_prompt_template = """\
You are an AI assistant capable of answering questions based on specific context provided to you. Your task is to retrieve relevant information from the context and generate a response. However, if the context is unrelated to the query or does not provide enough relevant information, you should respond with the following:
    • “I’m sorry, I cannot answer this question based on the provided context.”
    • “This topic is not covered by the provided information. Could you please clarify or provide more relevant context?”

Instructions:
    1. Contextual Relevance: Only provide an answer if the context directly relates to the query.
    2. Unrelated Context: If the context does not answer the question or is unrelated, do not attempt to fabricate an answer.
    3. Exactness: Make sure your response is directly drawn from the context provided. Do not rely on outside knowledge or assumptions.

Context:
{context}

Question:
{question}
"""

base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)

base_llm = ChatOpenAI(model="gpt-4o-mini", tags=["base_llm"])

retrieval_augmented_qa_chain = (
    # "question" and "context" are populated by chaining the question into the retriever
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    # Context is assigned to a RunnablePassthrough object
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # Format the prompt and generate response with the LLM
    | {"response": base_rag_prompt | base_llm, "context": itemgetter("context")}
)

response = retrieval_augmented_qa_chain.invoke({"question": "What is LangSmith?"}, {"tags": ["Demo Run"]})

print(response['response'].content)