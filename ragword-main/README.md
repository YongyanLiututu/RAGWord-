# RAG_Word: Retrieval-Augmented Generation for Word Documents

**RAG_Word** is an advanced AI-powered system for processing `.docx` Word documents, enabling intelligent question-answering through context-aware retrieval and generative language models. This project combines state-of-the-art technologies to build a seamless pipeline for extracting, indexing, and retrieving information from large volumes of unstructured textual data.

** NEW: Knowledge Graph Enhancement Module** - Now featuring advanced entity extraction, relationship mapping, and graph-based retrieval for enhanced accuracy and contextual understanding!

---

## ✨ Features

### Automated Document Handling
- Load and process multiple Word documents (`.docx`) with robust error handling.
- Dynamically splits large documents into manageable chunks for better performance.

### Vectorized Storage & Retrieval
- Integrates with **Chroma**, a high-performance vector database.
- Uses similarity-based indexing for precise and efficient retrieval.

### Intelligent Querying
- Combines retrieval and language models to provide concise and relevant answers.
- Adapts to various query styles and ensures high contextual accuracy.
- **NEW: Knowledge Graph-based retrieval** for entity and relationship-aware queries.

### Knowledge Graph Enhancement
- **Entity Extraction:** Automatically identifies persons, organizations, locations, dates, and concepts.
- **Relationship Mapping:** Discovers connections between entities (works for, located in, contains, etc.).
- **Graph-based Retrieval:** Leverages NetworkX for efficient graph traversal and querying.
- **Consistency Verification:** Validates generated answers against knowledge graph facts.

### Scalable & Modular Design
- Easily handles large document repositories.
- Modular architecture for customization and future enhancements.
- **NEW: Hybrid retrieval system** combining vector search, BM25, and knowledge graph retrieval.

---

## 📚 Background

Organizations frequently deal with vast amounts of unstructured data, making it challenging to extract actionable insights. **RAG_Word** addresses this problem by leveraging machine learning and retrieval techniques to:
- **Streamline data access:** Quickly locate the most relevant information.
- **Improve productivity:** Enable users to interact with documents in natural language.
- **Leverage modern AI:** Employ the latest advancements in embeddings and generative AI.

---

## 🛠️ Technologies

### Frameworks and Libraries

| Technology       | Purpose                                        |
|-------------------|------------------------------------------------|
| **LangChain**     | Orchestrates RAG pipeline components.         |
| **Mistral AI**    | Generates embeddings and processes queries.   |
| **Chroma**        | Vector storage for efficient similarity search.|
| **python-docx**   | Extracts and preprocesses Word document content. |
| **dotenv**        | Manages secure access to environment variables.|
| **NetworkX**      | Knowledge graph construction and traversal.   |
| **spaCy**         | Advanced NLP for entity and relation extraction.|

### Key Concepts
- **Retrieval-Augmented Generation (RAG):** Combines document retrieval with generative AI for contextual question answering.
- **Recursive Text Splitting:** Divides documents into smaller, non-overlapping chunks to optimize embedding accuracy.
- **Similarity Search:** Retrieves top-matching chunks using vectorized cosine similarity.
- **Knowledge Graph (KG):** Structured representation of entities and relationships extracted from documents.
- **Hybrid Retrieval:** Combines vector search, BM25 keyword matching, and graph-based retrieval for comprehensive results.
- **Multi-Agent System:** Coordinated agents for document processing, retrieval, reasoning, and answer generation.

---

## 📂 Directory Structure

```plaintext
├── main.py                    # Entry point for the pipeline
├── enhanced_rag_system.py     # Enhanced RAG system with KG integration
├── word_loader.py             # Module for Word document loading and processing
├── enhanced_word_loader.py    # Enhanced document processing with chunking
├── hybrid_retriever.py        # Hybrid retrieval (vector + BM25 + KG)
├── kg_loader.py               # Knowledge graph construction from documents
├── kg_retriever.py            # Knowledge graph querying and retrieval
├── enhanced_agents.py         # Multi-agent system including KGAwareAgent
├── layered_generator.py       # Layered answer generation with KG verification
├── conversation_manager.py    # Multi-turn conversation management
├── requirements.txt           # Python dependencies
├── requirements_kg.txt        # Additional dependencies for KG features
├── .env                       # Environment variables (e.g., Mistral API key)
├── chroma_db/                 # Directory for Chroma database persistence
├── faiss_index/               # FAISS vector index storage
├── README.md                  # Project documentation
├── README_KG_Enhancement.md   # Detailed KG enhancement documentation
└── quick_start_kg.md          # Quick start guide for KG features
```

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- Pip (Python package installer)

### Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/rag_word.git
   cd rag_word

    ```

## Install Dependencies

### Basic Installation
Run the following command to install core dependencies:

```bash
pip install -r requirements.txt
```

### Knowledge Graph Enhancement
For enhanced features including knowledge graph capabilities:

```bash
pip install -r requirements_kg.txt
```

### Quick Start with KG Features
See `quick_start_kg.md` for detailed setup and usage instructions.

## 🚀 Quick Start

### Basic Usage
```python
from enhanced_rag_system import EnhancedRAGSystem

# Initialize the system
rag_system = EnhancedRAGSystem(
    model_name="gpt-4",
    enable_enhanced_features=True,
    enable_kg=True,  # Enable knowledge graph features
    kg_mode="fusion"  # Use KG fusion mode
)

# Upload documents
documents = ["path/to/document1.docx", "path/to/document2.docx"]
result = rag_system.upload_documents(documents)

# Ask questions
answer = rag_system.ask_question("What is the main topic discussed in the documents?")
print(answer['answer'])
```

### Knowledge Graph Features
```python
# Get KG statistics
kg_stats = rag_system.get_kg_statistics()
print(f"Entities: {kg_stats['entities_count']}")
print(f"Relationships: {kg_stats['relationships_count']}")

# Search entities in KG
entity_info = rag_system.search_entity_in_kg("John Smith", "PERSON")
print(entity_info)

# Export knowledge graph
kg_data = rag_system.export_knowledge_graph()
```

## 📖 Documentation

- **[README_KG_Enhancement.md](README_KG_Enhancement.md)** - Detailed documentation for knowledge graph features
- **[quick_start_kg.md](quick_start_kg.md)** - Quick start guide for KG enhancement
- **[test_kg_system.py](test_kg_system.py)** - Test suite for KG functionality

## 🔧 Configuration

The system supports various configuration options:

```python
# Basic configuration
rag_system = EnhancedRAGSystem(
    model_name="gpt-4",           # LLM model
    emb_model="text-embedding-ada-002",  # Embedding model
    use_mistral=False,            # Use Mistral AI models
    enable_enhanced_features=True, # Enable advanced features
    enable_kg=True,               # Enable knowledge graph
    kg_mode="fusion"              # KG mode: "fusion" or "verify"
)

# Configure KG settings
rag_system.set_kg_config(
    enable_kg=True,
    kg_weight=0.3  # Weight for KG results in fusion
)
```

## 🎯 Use Cases

### Enhanced Document Analysis
- **Entity Recognition:** Automatically identify people, organizations, and locations
- **Relationship Discovery:** Find connections between entities across documents
- **Contextual Understanding:** Better comprehension of document structure and meaning

### Intelligent Question Answering
- **Multi-modal Retrieval:** Combine vector search, keyword matching, and graph traversal
- **Fact Verification:** Validate answers against knowledge graph facts
- **Relationship Queries:** Answer questions about entity relationships

### Research and Analysis
- **Document Mining:** Extract structured information from unstructured text
- **Knowledge Discovery:** Uncover hidden patterns and connections
- **Data Validation:** Ensure consistency across multiple documents

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **Mistral AI** for advanced language models
- **NetworkX** for knowledge graph implementation
- **spaCy** for NLP capabilities

