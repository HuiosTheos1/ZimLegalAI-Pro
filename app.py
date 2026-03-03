import streamlit as st
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from engine import initialize_brain

st.set_page_config(page_title="Zim-Legal AI", page_icon="⚖️")
st.title("⚖️ Zim-Legal AI")

# We will put your free API key in the "Secrets" settings we talked about
if 'db' not in st.session_state:
    with st.spinner("Loading Zimbabwean Statutes..."):
        st.session_state.db = initialize_brain()

if st.session_state.db:
    # Use Groq for lightning speed on Android
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=st.secrets["GROQ_API_KEY"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=st.session_state.db.as_retriever()
    )

    query = st.text_input("Ask a legal question:")
    if st.button("Consult AI"):
        res = qa_chain.invoke(query)
        st.write(res["result"])
