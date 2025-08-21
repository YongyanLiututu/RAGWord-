from rag_faiss import RAG_Word_FAISS

if __name__ == "__main__":
    rag = RAG_Word_FAISS()
    rag.read_directory("path/to/word_files")
    result = rag.ask("What is LLM?")
    print("Input:", result["input"])
    print("Context:", [doc["page_content"][:100] for doc in result["context"]])
    print("Answer:", result["answer"])
    rag.clear()
