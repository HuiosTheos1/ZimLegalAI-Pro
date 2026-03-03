import streamlit as st
import os

# 1. The "Classic" Chain (Required for RetrievalQA in 2026)
from langchain_classic.chains import RetrievalQA 

# 2. The Cloud Brain (Fastest for Android)
from langchain_groq import ChatGroq 

# 3. Handling the Laws (PDFs & Vector Database)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 4. Your custom engine (Assuming it's in the same folder)
from engine import initialize_brain
