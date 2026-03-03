import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CLYTON MAKATE'S LEGAL BRAIN ENGINE ---

@st.cache_resource  # 🚀 THIS IS THE SPEED BOOSTER
def initialize_brain():
    """
    This function reads your 'docs/' folder, splits the laws into 
    searchable chunks, and stores them in a lightning-fast vector database.
    """
    # 1. Load all legal PDFs from your 'docs' folder
    # Ensure you have uploaded your PDFs to a folder named 'docs' on GitHub
    loader = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # 2. Split the law into small, readable chunks for the AI
    # We use 1000 characters so the AI gets enough 'context' for each law
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 3. Use HuggingFace to turn text into 'Numbers' (Embeddings)
    # This is free and extremely fast for Zimbabwean Statutes
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create the Searchable Database (FAISS)
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore
