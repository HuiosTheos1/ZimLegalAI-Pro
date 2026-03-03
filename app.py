import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. THE LEGAL BRAIN ENGINE ---
@st.cache_resource
def initialize_brain():
    if not os.path.exists('./docs/'):
        os.makedirs('./docs/')
    
    loader = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        st.error("⚠️ No legal PDFs found in the 'docs/' folder!")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

# --- 2. CONFIG & DISCLAIMER ---
st.set_page_config(page_title="Zim-Legal AI Master", page_icon="⚖️", layout="wide")

if "disclaimer_accepted" not in st.session_state:
    st.title("⚖️ Zim-Legal AI: Important Notice")
    st.warning("Founder: Clyton Makate. Please accept terms to proceed.")
    st.markdown("""
    1. **Not Legal Advice:** Educational tool for pre-trial preparation.
    2. **Verification:** Always verify AI results with the official Government Gazette.
    3. **Liability:** Clyton Makate is not liable for any legal outcomes.
    """)
    if st.button("I Accept these Terms & Conditions"):
        st.session_state.disclaimer_accepted = True
        st.rerun()
    st.stop()

# --- 3. INTAKE & PERSONALIZATION ---
if "user_name" not in st.session_state:
    st.title("⚖️ Initial Intake")
    with st.form("intake_form"):
        name = st.text_input("Full Name")
        lang = st.selectbox
