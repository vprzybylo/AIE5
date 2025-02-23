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
[Grid Code PDF](https://www.nationalgrid.com/sites/default/files/documents/8589935310-Complete%20Grid%20Code.pdf)

### Chunking Strategy
We employ a hybrid chunking approach:
- Section-based chunks for maintaining regulatory context
- Smaller overlapping chunks for detailed technical specifications

**chunk size**: The maximum size of each chunk is set to 2000 characters.

**chunk_overlap**: Each chunk overlaps with the previous chunk by 50 characters to ensure context continuity.

**separators**: The text is split based on the following separators in order of priority: double newlines (\n\n), single newline (\n), period (.), space ( ), and empty string ("").

## Task 4: End-to-End Prototype

[Link to Hugging Face Space]

## Task 5: Evaluation Dataset

- With out-of-the-box RAG:

| Metric               | Value   |
|----------------------|---------|
| Faithfulness          | 0.7958  |
| Answer Relevancy      | 0.8701  |
| Context Recall        | 0.9583  |
| Context Precision      | 0.8667  |


1. **Faithfulness (0.7958)**: This metric indicates that approximately 79.58% of the generated responses accurately reflect the ground truth. While this is a relatively good score, there is still room for improvement to ensure that the responses are more closely aligned with the expected outputs.

2. **Answer Relevancy (0.8701)**: With a score of 87.01%, the responses are highly relevant to the questions posed. This suggests that the model is effectively understanding and addressing the user's queries, which is a positive indicator of the pipeline's performance.

3. **Context Recall (0.9583)**: A context recall of 95.83% indicates that the model is very effective at retrieving relevant context from the documents. This high score suggests that the model is capable of accessing and utilizing the necessary information to generate responses.

4. **Context Precision (0.8667)**: With a context precision of 86.67%, the model is also good at ensuring that the contexts it retrieves are relevant to the user's questions. This score indicates that the model is not only retrieving a lot of relevant context but is also filtering out irrelevant information effectively.


## Task 6: Fine-Tuning Plan

- Generate synthetic Q&A pairs from Grid Code
- Fine-tune embedding model for technical vocabulary
- Optimize for regulatory compliance accuracy

## Task 7: Performance Assessment

Comparative metrics will be tracked for:
- Response accuracy
- Context relevance
- Query response time




