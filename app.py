import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated for 2026
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. THE LEGAL BRAIN ENGINE ---
@st.cache_resource
def initialize_brain():
    # Create docs folder if missing
    if not os.path.exists('./docs/'):
        os.makedirs('./docs/')
    
    # Load Laws from the docs folder
    loader = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        st.error("⚠️ No legal PDFs found in the 'docs/' folder! Please upload the Constitution or Labour Act.")
        st.stop()

    # Split law into chunks for the AI to "read"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create the searchable database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

# --- 2. CONFIG & DISCLAIMER ---
st.set_page_config(page_title="Zim-Legal AI Master", page_icon="⚖️", layout="wide")

if "disclaimer_accepted" not in st.session_state:
    st.title("⚖️ Zim-Legal AI: Important Notice")
    st.warning(f"Founder: Clyton Makate. Please accept terms to proceed.")
    st.markdown("""
    1. **Not Legal Advice:** This is an educational tool for pre-trial preparation.
    2. **Verification:** Always verify AI results with the official Government Gazette.
    3. **Liability:** Clyton Makate is not liable for any legal outcomes or decisions.
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
        lang = st.selectbox("Preferred Language", ["English", "Shona", "Ndebele", "Mixed"])
        role = st.selectbox("I am an:", ["Accused Person", "Employee", "Employer", "Lawyer", "Student"])
        if st.form_submit_button("Start Legal Strategy"):
            if name:
                st.session_state.update({
                    "user_name": name, 
                    "user_lang": lang, 
                    "user_role": role, 
                    "score": 50
                })
                st.rerun()
            else: st.error("Name is required.")
    st.stop()

# --- 4. MAIN APP INTERFACE ---
st.title(f"⚖️ {st.session_state.user_name}'s Pre-Trial Session")

# Sidebar
with st.sidebar:
    st.header("📊 Readiness")
    st.metric("Success Probability", f"{st.session_state.score}%")
    st.progress(st.session_state.score / 100)
    st.divider()
    if st.button("🚨 EMERGENCY: ARRESTED", type="primary"): 
        st.session_state.emergency = True

# Emergency Mode Overlays
if st.session_state.get('emergency'):
    st.
