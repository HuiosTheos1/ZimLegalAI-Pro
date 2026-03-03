import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def initialize_brain():
    # Path where your PDFs are stored on GitHub
    docs_path = "docs/"
    
    if not os.path.exists(docs_path):
        return None

    all_docs = []
    # Loop through every PDF you uploaded
    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_path, file))
            all_docs.extend(loader.load())

    if not all_docs:
        return None

    # Split long laws into smaller, readable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)

    # Create the search index using HuggingFace (Free & Cloud Compatible)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    
    return db
