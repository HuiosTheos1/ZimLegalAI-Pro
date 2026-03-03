import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_brain():
    docs_path = "docs/"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
    
    docs = []
    for f in os.listdir(docs_path):
        if f.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_path, f))
            docs.extend(loader.load())
    
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # FAISS is much more stable on Python 3.14
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore