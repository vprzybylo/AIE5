import numpy as np
from aimakerspace.text_utils import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
import asyncio
from aimakerspace.vectordatabase import VectorDatabase
from rich import print as rprint


# Load PDF
loader = PyPDFLoader("data/test.pdf")
documents = loader.load()

#rprint(documents)
"""
using PyPDFLoader we get metadata and page_content:

Document(
    metadata={'source': 'data/test.pdf', 'page': 45},
    page_content='...'
)

"""

# Split text
text_splitter = CharacterTextSplitter()
split_documents = text_splitter.split_texts(documents)


RAG_PROMPT_TEMPLATE = """ \
Use the provided context to answer the user's query.

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".
"""

rag_prompt = SystemRolePrompt(RAG_PROMPT_TEMPLATE)

USER_PROMPT_TEMPLATE = """ \
Context:
{context}

User Query:
{user_query}
"""


user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)


class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI(), vector_db_retriever: VectorDatabase) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever

    def run_pipeline(self, user_query: str) -> str:
        context_list = self.vector_db_retriever.search_by_text(user_query, k=4)  # DISTANCE MEASURE UPDATED IN vectordatabse.py
        context_prompt = ""
        for context in context_list:
            context_prompt += context[0] + "\n"

        formatted_system_prompt = rag_prompt.create_message()

        formatted_user_prompt = user_prompt.create_message(user_query=user_query, context=context_prompt)

        return {"response": self.llm.run([formatted_system_prompt, formatted_user_prompt]), "context": context_list}


vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents)) 

retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=ChatOpenAI()
)

response = retrieval_augmented_qa_pipeline.run_pipeline("What is chain of thought reasoning'?")
rprint(response)