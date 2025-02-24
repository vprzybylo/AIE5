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

- 🔢 **Embedding Models**: text-embedding-3-small, Snowflake/snowflake-arctic-embed-l, [finetuned-arctic-ft on huggingface](https://huggingface.co/vanessaprzybylo/finetuned_arctic_ft)
*Starting with OpenAI for rapid prototyping, then fine-tuning snowflake-arctic-embed-l for domain-specific understanding*

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

## Embedding Model Comparison

### OpenAI text-embedding-3-small
| Metric               | Value   |
|----------------------|---------|
| Faithfulness         | 0.9056  |
| Answer Relevancy     | 0.6007  |
| Context Recall       | 0.7955  |
| Context Precision    | 0.5833  |

### Fine-tuned snowflake-arctic-embed-l
| Metric               | Value   |
|----------------------|---------|
| Faithfulness         | 0.8045  |
| Answer Relevancy     | 0.5738  |
| Context Recall       | 0.5556  |
| Context Precision    | 0.4600  |

### Analysis

The comparison reveals several interesting findings:

1. **OpenAI Superiority**: OpenAI's text-embedding-3-small outperforms the fine-tuned model across all metrics:
   - Higher faithfulness (90.56% vs 80.45%)
   - Better answer relevancy (60.07% vs 57.38%)
   - Stronger context recall (79.55% vs 55.56%)
   - Better context precision (58.33% vs 46.00%)

2. **Fine-tuning Impact**: Despite being fine-tuned on technical documentation, the snowflake-arctic-embed-l model showed:
   - ~10% decrease in faithfulness
   - ~24% decrease in context recall
   - ~12% decrease in context precision

3. **Potential Factors**:
   - The base model (snowflake-arctic-embed-l) may not be as sophisticated as OpenAI's embedding model
   - The fine-tuning dataset or process might need optimization or more data (only passed 20 pages of the Grid Code)
   - OpenAI's model may have better pre-training on technical documentation
   - Rate limiting of the API calls to OpenAI may have affected the results


## Task 6: Fine-Tuning Plan

- Generate synthetic Q&A pairs from Grid Code
- Fine-tune embedding model for technical vocabulary
- Optimize for regulatory compliance accuracy

## Task 7: Performance Assessment

Comparative metrics will be tracked for:
- Response accuracy
- Context relevance
- Query response time




