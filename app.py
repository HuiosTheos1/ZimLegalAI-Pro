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
        st.error("⚠️ No legal PDFs found in the 'docs/' folder! Please upload the Constitution or Labour Act.")
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
        st.write("Enter your details to begin the simulation.")
        name = st.text_input("Full Name")
        lang = st.selectbox("Preferred Language", ["English", "Shona", "Ndebele", "Mixed"])
        role = st.selectbox("I am an:", ["Accused Person", "Employee", "Employer", "Lawyer", "Student"])
        
        submitted = st.form_submit_button("Start Legal Strategy")
        
        if submitted:
            if name:
                st.session_state.update({
                    "user_name": name, 
                    "user_lang": lang, 
                    "user_role": role, 
                    "score": 50
                })
                st.rerun()
            else:
                st.error("Name is required.")
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
    st.error("### 🔴 CONSTITUTIONAL PROTOCOL (Sec 50)")
    st.markdown("1. **Remain Silent.**\n2. **Demand a Lawyer.**\n3. **48-Hour Rule.**")
    if st.button("Return to Session"): 
        st.session_state.emergency = False
        st.rerun()
    st.stop()

# Load Brain
db = initialize_brain()

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Hello {st.session_state.user_name}. I am your Teacher. Describe your legal issue."}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- 5. THE INTERROGATION ---
if prompt := st.chat_input("Explain what happened..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=st.secrets["GROQ_API_KEY"])
    
    TEMPLATE = f"""You are the 'Zim-Legal Pre-Trial Teacher' for {st.session_state.user_name}.
    ROLE: {st.session_state.user_role} | LANGUAGE: {st.session_state.user_lang}
    CONTEXT: {{context}} | HISTORY: {{chat_history}}
    
    1. Act as a tough Prosecutor challenging the user's story.
    2. Warn if they say something damaging: "⚠️ STOP! {st.session_state.user_name}..."
    3. End EVERY reply with 'Readiness Score: X/100'.
    QUESTION: {{question}}"""
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=db.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=TEMPLATE, input_variables=["context", "chat_history", "question"])}
    )
    
    with st.chat_message("assistant"):
        res = qa({"question": prompt, "chat_history": st.session_state.chat_history})
        ans = res["answer"]
        
        # Logic to update the score meter
        if "Readiness Score:" in ans:
            try:
                score_str = ans.split("Readiness Score:")[1].split("/")[0].strip()
                st.session_state.score = int(score_str)
            except:
                pass
            
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.session_state.chat_history.append((prompt, ans))
