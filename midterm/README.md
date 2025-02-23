# Grid Code Assistant: An Intelligent RAG-based Chatbot for Field Workers

## Background and Context

Field workers in the electricity transmission sector face challenges in quickly accessing and interpreting the Grid Code - a complex technical document that governs the development, maintenance, and operation of the electricity transmission system. The Grid Code Assistant aims to make this critical information more accessible through an intelligent chatbot interface.

## Task 1: Problem Definition and Audience

### Problem Statement
Field workers need immediate, accurate answers to Grid Code questions while working on electricity transmission systems, but the document's complexity and length make quick reference difficult.

### User Impact
Our primary users are field workers who:
- Need to make time-critical decisions based on Grid Code requirements
- May have limited time to search through lengthy documentation
- Require accurate technical information to ensure compliance and safety
- Often work in conditions where accessing and reading detailed documentation is impractical

## Task 2: Proposed Solution

We will build a RAG-based chatbot with the following stack:

- 🤖 **LLM**: GPT-4 
*Chosen for its strong technical understanding and ability to process complex regulatory language*

- 🔢 **Embedding Models**: text-embedding-3-small, BAAI/bge-large-en-v1.5
*Starting with OpenAI for rapid prototyping, then fine-tuning BAAI for domain-specific understanding*

- 🎺 **Orchestration**: LangChain
*Provides robust RAG pipeline components and easy integration with evaluation tools*

- ↗️ **Vector Store**: Qdrant
*Efficient similarity search and excellent support for metadata filtering*

- 📈 **Monitoring**: LangSmith
*Comprehensive tracing and evaluation capabilities for RAG systems*

- 📐 **Evaluation**: RAGAS
*Industry standard for evaluating RAG system performance*

- 💬 **User Interface**: Streamlit
*Fast deployment and simple interface for field workers*

- 🛎️ **Deployment**: Hugging Face Spaces
*Reliable hosting with good uptime and easy deployment*

## Task 3: Data Strategy

### Data Sources
![Grid Code PDF](https://www.nationalgrid.com/sites/default/files/documents/8589935310-Complete%20Grid%20Code.pdf)

### Chunking Strategy
We employ a hybrid chunking approach:
- Section-based chunks for maintaining regulatory context
- Smaller overlapping chunks (500 tokens) for detailed technical specifications
- Special handling for tables and diagrams

## Task 4: End-to-End Prototype

[Link to Hugging Face Space]

## Task 5: Evaluation Dataset

Initial RAGAS metrics:
| Metric | Target Value |
|--------|--------------|
| Context Recall |  |
| Faithfulness | |
| Answer Relevancy | |

## Task 6: Fine-Tuning Plan

- Generate synthetic Q&A pairs from Grid Code
- Fine-tune embedding model for technical vocabulary
- Optimize for regulatory compliance accuracy

## Task 7: Performance Assessment

Comparative metrics will be tracked for:
- Response accuracy
- Context relevance
- Query response time
- User satisfaction
