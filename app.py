import streamlit as st
import os

# 2026 Imports: Using the 'classic' package for backward compatibility
from langchain_classic.chains import RetrievalQA 
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import your own local logic
from engine import initialize_brain

# Set up the Android Interface
st.set_page_config(page_title="Zim-Legal AI", page_icon="⚖️")
st.title("⚖️ Zim-Legal AI")
st.markdown("### Zimbabwean Law Intelligence")

# Connect to the Groq Cloud Key (Setup in Streamlit Secrets)
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("Missing GROQ_API_KEY in Secrets!")
    st.stop()

# Initialize or retrieve the database
if 'db' not in st.session_state:
    with st.spinner("Analyzing Zimbabwean Statutes..."):
        st.session_state.db = initialize_brain()

if st.session_state.db:
    # High-speed 2026 Cloud Engine (Llama 3)
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=st.session_state.db.as_retriever()
    )

    query = st.text_input("Ask a legal question (e.g., 'What are the mining rights under the Act?')")
    
    if st.button("Consult AI"):
        with st.spinner("Searching laws..."):
            res = qa_chain.invoke(query)
            st.markdown("---")
            st.markdown("#### 📜 Legal Response")
            st.write(res["result"])
