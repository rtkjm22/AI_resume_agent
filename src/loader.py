from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_vectorstore(filepath: str, persist_path: str = "./faiss_index"):
    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=512,
      chunk_overlap=50,
    )
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(split_docs, embeddings)
    vectordb.save_local(persist_path)