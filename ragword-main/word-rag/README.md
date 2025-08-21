# RAG_Word: Retrieval-Augmented Generation for Word Documents

**RAG_Word** is an advanced AI-powered system for processing `.docx` Word documents, enabling intelligent question-answering through context-aware retrieval and generative language models. This project combines state-of-the-art technologies to build a seamless pipeline for extracting, indexing, and retrieving information from large volumes of unstructured textual data.

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

### Scalable & Modular Design
- Easily handles large document repositories.
- Modular architecture for customization and future enhancements.

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

### Key Concepts
- **Retrieval-Augmented Generation (RAG):** Combines document retrieval with generative AI for contextual question answering.
- **Recursive Text Splitting:** Divides documents into smaller, non-overlapping chunks to optimize embedding accuracy.
- **Similarity Search:** Retrieves top-matching chunks using vectorized cosine similarity.

---

## 📂 Directory Structure

```plaintext
├── main.py                # Entry point for the pipeline
├── word_loader.py         # Module for Word document loading and processing
├── rag_word.py            # Core logic for RAG pipeline and query answering
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (e.g., Mistral API key)
├── chroma_db/             # Directory for Chroma database persistence
└── README.md              # Project documentation
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

Run the following command to install dependencies:

```bash
pip install -r requirements.txt

 ```

