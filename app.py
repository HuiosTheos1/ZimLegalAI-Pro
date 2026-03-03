import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from engine import initialize_brain

st.set_page_config(page_title="Zim-Legal AI", page_icon="⚖️")
st.title("⚖️ Zim-Legal AI")

if 'db' not in st.session_state:
    with st.spinner("Initializing Zimbabwean Law Database..."):
        st.session_state.db = initialize_brain()

if st.session_state.db is None:
    st.error("⚠️ No PDFs found! Drop your Zimbabwean Law PDFs into the 'docs' folder and refresh.")
else:
    llm = OllamaLLM(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=st.session_state.db.as_retriever()
    )

    user_input = st.text_input("Ask a legal question:")
    if st.button("Consult AI"):
        with st.spinner("Analyzing statutes..."):
            response = qa_chain.invoke(user_input)
            st.markdown("### 📜 AI Legal Guidance")
            st.write(response["result"])