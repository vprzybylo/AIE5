from aimakerspace.text_utils import CharacterTextSplitter, TextFileLoader
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
import asyncio
from aimakerspace.vectordatabase import VectorDatabase
from rich import print as rprint
from datetime import datetime
import os


# Load PDF
text_loader = TextFileLoader("data/test.pdf")
documents = text_loader.load_documents()


text_splitter = CharacterTextSplitter()
split_documents = text_splitter.split_texts(documents)

# Enhance documents with metadata and modify content
enhanced_documents = []
for i, doc in enumerate(split_documents):

    # Extract original document info
    original_metadata = doc.metadata
    
    # Enhanced metadata
    metadata = {
        # Document identification
        "doc_id": f"doc_{os.path.basename(original_metadata.get('source', 'unknown'))}_{i}",
        "source": original_metadata.get('source', 'unknown'),
        "page": original_metadata.get('page', 'unknown'),
        "filename": original_metadata.get('filename', 'unknown'),
        "date_added": original_metadata.get('date_added', 'unknown'),
        
        # Content classification
        "content_type": "research_paper",
        "topic": original_metadata.get('topic', 'LLM'),
    }
    
    # Add metadata header to content for embedding
    enhanced_content = f"""
Source: Page {metadata['page']} of {metadata['source']}
Content Type: {metadata['content_type']}
Topic: {metadata['topic']}
Date Added: {metadata['date_added']}

{doc.page_content}
"""
    
    # Create new document with enhanced content and metadata
    doc.page_content = enhanced_content
    doc.metadata.update(metadata)
    enhanced_documents.append(doc)

# Update split_documents with enhanced versions
split_documents = enhanced_documents
#rprint(split_documents)

RAG_PROMPT_TEMPLATE = """ \
Use the provided context to answer the user's query.

You may not answer the user's query unless there is specific context in the following text.

If the answer is not provided in the text please respond with "I don't know".
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
        results = self.vector_db_retriever.search_by_text(user_query, k=4)
        context_prompt = ""
        
        # Collect metadata from retrieved contexts
        metadata_info = []
        for doc, score in results:
            context_prompt += doc.page_content + "\n"
            metadata_info.append({
                'doc_id': doc.metadata.get('doc_id', 'unknown'),
                'page': doc.metadata.get('page', 'unknown'),
                'section': doc.metadata.get('section', 'unknown'),
                'topic': doc.metadata.get('topic', 'unknown'),
                'score': score
            })

        formatted_system_prompt = rag_prompt.create_message()
        formatted_user_prompt = user_prompt.create_message(
            user_query=user_query, 
            context=context_prompt
        )
        
        return {
            "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]),
            "context": [doc for doc, _ in results],
            "metadata": metadata_info
        }


vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents)) 

retrieval_augmented_qa_pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=ChatOpenAI()
)
question1 = 'Talk about LLMs for education'
response = retrieval_augmented_qa_pipeline.run_pipeline(question1)
rprint('question', question1)
rprint('response', response['response'])
rprint('\n\n')

question2 = 'What is the content type of the LLM paper and what date was it added?'
response = retrieval_augmented_qa_pipeline.run_pipeline(question2)
rprint('question', question2)
rprint(response['response'])