import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. THE LEGAL BRAIN ENGINE (Now integrated) ---
@st.cache_resource
def initialize_brain():
    if not os.path.exists('./docs/'):
        os.makedirs('./docs/')
    
    # Load Laws
    loader = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        st.error("No legal PDFs found in the 'docs/' folder!")
        st.stop()

    # Split & Embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return FAISS.from_documents(texts, embeddings)

# --- 2. THE APP INTERFACE ---
st.set_page_config(page_title="Zim-Legal AI Master", page_icon="⚖️", layout="wide")

# DISCLAIMER & INTAKE (Same logic as before)
if "disclaimer_accepted" not in st.session_state:
    st.title("⚖️ Zim-Legal AI: Important Notice")
    st.warning("By Clyton Makate. Accept terms to proceed.")
    st.markdown("1. Not Legal Advice. 2. Human-in-the-loop. 3. No Liability.")
    if st.button("I Accept terms"):
        st.session_state.disclaimer_accepted = True
        st.rerun()
    st.stop()

if "user_name" not in st.session_state:
    st.title("⚖️ Initial Intake")
    with st.form("intake"):
        name = st.text_input("Full Name")
        lang = st.selectbox("Language", ["English", "Shona", "Ndebele"])
        role = st.selectbox("Role", ["Accused", "Employee", "Employer", "Lawyer"])
        if st.form_submit_button("Start Session"):
            if name:
                st.session_state.update({"user_name": name, "user_lang": lang, "user_role": role, "score": 50})
                st.rerun()
    st.stop()

# --- 3. MAIN INTERROGATION LOGIC ---
st.title(f"⚖️ {st.session_state.user_name}'s Pre-Trial Session")

# Sidebar
with st.sidebar:
    st.header("📊 Readiness")
    st.metric("Success Probability", f"{st.session_state.score}%")
    if st.button("🚨 EMERGENCY", type="primary"): st.session_state.emergency = True

if st.session_state.get('emergency'):
    st.error("### 🔴 SEC 50 PROTOCOL: Remain Silent. Demand a Lawyer. 48-Hour Rule.")
    if st.button("Back"): 
        st.session_state.emergency = False
        st.rerun()
    st.stop()

# Load Brain
db = initialize_brain()

# Chat logic
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Explain your case so we can begin."}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("State your defense..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=st.secrets["GROQ_API_KEY"])
    
    TEMPLATE = f"""You are the 'Zim-Legal Pre-Trial Teacher' for {st.session_state.user_name} ({st.session_state.user_role}).
    Respond in {st.session_state.user_lang}. Context: {{context}}. Warn if they make mistakes.
    End with 'Readiness Score: X/100'.
    Question: {{question}}"""
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=db.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])}
    )
    
    with st.chat_message("assistant"):
        res = qa({"question": prompt, "chat_history": st.session_state.chat_history})
        ans = res["answer"]
        if "Readiness Score:" in ans:
            try: st.session_state.score = int(ans.split("Readiness Score:")[1].split("/")[0].strip())
            except: pass
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.session_state.chat_history.append((prompt, ans))
