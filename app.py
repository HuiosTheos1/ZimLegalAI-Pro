import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from engine import initialize_brain

st.set_page_config(page_title="Zim-Legal AI", page_icon="⚖️")
st.title("⚖️ Zim-Legal AI")

# Use Streamlit Secrets for your API Key (I'll show you how next)
groq_api_key = st.secrets["GROQ_API_KEY"]

if 'db' not in st.session_state:
    with st.spinner("Processing Zimbabwean Law Database..."):
        st.session_state.db = initialize_brain()

if st.session_state.db:
    # This Llama3 engine runs on Groq's high-speed servers
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=st.session_state.db.as_retriever()
    )

    query = st.text_input("Ask a legal question (e.g., 'What are my rights in the 2013 Constitution?'):")
    if st.button("Consult AI"):
        with st.spinner("Analyzing statutes..."):
            res = qa_chain.invoke(query)
            st.markdown("### 📜 AI Legal Guidance")
            st.write(res["result"])
