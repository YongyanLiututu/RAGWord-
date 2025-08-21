import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from word_loader import WordLoader
import time

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY", "aip")

class RAG_Word_FAISS:
    def __init__(self, model_name="gpt-4", emb_model="text-embedding-ada-002"):
        self.model = ChatOpenAI(model=model_name, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(model=emb_model, openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        self.system_prompt = (
            "You are an assistant for answering questions based on the given context. "
            "If you don't know the answer, simply say 'I don't know.' Keep the response concise."
        )
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None

    def read_directory(self, directory_path: str):
        faiss_index_file = os.path.join(directory_path, "faiss_index")
        if os.path.exists(faiss_index_file):
            self.load_existing_faiss_index(faiss_index_file)
        else:
            self.build_new_faiss_index(directory_path, faiss_index_file)

    def load_existing_faiss_index(self, faiss_index_file):
        self.vector_store = FAISS.load_local(faiss_index_file, self.embeddings)
        self.initialize_retriever_and_chain()

    def build_new_faiss_index(self, directory_path: str, faiss_index_file: str):
        word_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.docx')]
        all_docs = self.load_all_documents(word_files)
        chunks = self.split_documents_into_chunks(all_docs)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(faiss_index_file)
        self.initialize_retriever_and_chain()

    def load_all_documents(self, word_files):
        all_docs = []
        for file in word_files:
            loader = WordLoader(file)
            docs = loader.load()
            all_docs.extend(docs)
        return all_docs

    def split_documents_into_chunks(self, documents):
        return self.text_splitter.split_documents(documents)

    def initialize_retriever_and_chain(self):
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.model, retriever=self.retriever, return_source_documents=True
        )

    def ask(self, query: str):
        if not self.retriever:
            return "No Vector DB.", []
        retrieved_docs = self.retriever.get_relevant_documents(query)
        if not retrieved_docs:
            return {"input": query, "context": [], "answer": "No relevant documents found."}
        answer = self.invoke_with_retry(lambda: self.qa_chain.run(query))
        return {"input": query, "context": retrieved_docs, "answer": answer}

    def invoke_with_retry(self, func, max_retries=5, delay=10):
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                time.sleep(delay)
        raise Exception("Max retries exceeded.")

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
