import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. ENGINE ---
@st.cache_resource
def initialize_brain():
    if not os.path.exists('./docs/'): os.makedirs('./docs/')
    loader = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        st.error("⚠️ No Law PDFs found. Please upload to 'docs/' folder.")
        st.stop()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

# --- 2. SETUP ---
st.set_page_config(page_title="Zim-Legal AI Hub", page_icon="⚖️", layout="wide")

if "user_name" not in st.session_state:
    st.title("⚖️ Zim-Legal AI: Justice for All")
    st.info("Founder: Clyton Makate")
    name = st.text_input("Enter your name:")
    if st.button("Enter Dashboard"):
        if name:
            st.session_state.user_name = name
            st.session_state.score = 50
            st.rerun()
    st.stop()

# --- 3. DASHBOARD MENU ---
with st.sidebar:
    st.title(f"Hi, {st.session_state.user_name}")
    choice = st.radio("Main Menu:", [
        "🏠 Home (Our Mission)",
        "🏢 Retail & Store Compliance",
        "⛏️ Mining & Claims Law",
        "🚕 Transport & Taxi Regulations",
        "⚖️ Legal AI Advisor (Pre-Trial)",
        "📂 Document Library"
    ])
    st.divider()
    if choice == "⚖️ Legal AI Advisor (Pre-Trial)":
        st.metric("Readiness Score", f"{st.session_state.score}%")
        st.progress(st.session_state.score / 100)

# --- 4. THE PAGES ---

# --- HOME (NEW) ---
if choice == "🏠 Home (Our Mission)":
    st.title("⚖️ Zim-Legal AI")
    st.subheader("Motto: 'Justice Delayed is Justice Denied — Empowering the Zimbabwean Citizen.'")
    st.markdown("---")
    st.markdown("""
    ### 🛡️ Our Mission
    To bridge the gap between complex law and the everyday citizen. Whether you are a small business owner 
    operating a retail store, a taxi operator, or an employee seeking justice, **Zim-Legal AI** is your 
    first line of defense.
    
    ### 🚀 How to use this Dashboard:
    - **Business Owners:** Select your sector (Retail, Mining, Transport) to see 2026 compliance checklists.
    - **Legal Prep:** Use the **AI Advisor** to test your case before going to court or a hearing.
    - **Law Library:** Download official statutes to keep on your phone.
    """)
    st.success(f"Welcome to the hub, {st.session_state.user_name}. Select a tool in the sidebar to begin.")

# --- SECTORS ---
elif choice == "🏢 Retail & Store Compliance":
    st.header("Retail & Shop License Guide")
    st.write("Ensuring your store meets ZIMRA and Local Council standards.")

elif choice == "⛏️ Mining & Claims Law":
    st.header("Mining Compliance & EIA")
    st.write("For small-scale miners navigating the Ministry of Mines.")

elif choice == "🚕 Transport & Taxi Regulations":
    st.header("Public Service Vehicle (PSV) Rules")
    st.write("Stay compliant with VID and Road Traffic Act regulations.")

# --- AI ADVISOR (FIXED GROQ ERROR) ---
elif choice == "⚖️ Legal AI Advisor (Pre-Trial)":
    st.header("Legal AI Interrogator")
    db = initialize_brain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("State your legal issue..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=st.secrets["GROQ_API_KEY"])
        
        TEMPLATE = f"""You are the 'Zim-Legal Advisor' for {st.session_state.user_name}.
        Challenge the user's story, warn of legal risks, and give a 'Readiness Score: X/100' at the end.
        
        CONTEXT: {{context}} | HISTORY: {{chat_history}}
        QUESTION: {{question}}"""
        
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=db.as_retriever(),
            combine_docs_chain_kwargs={"prompt": PromptTemplate(template=TEMPLATE, input_variables=["context", "chat_history", "question"])}
        )
        
        with st.chat_message("assistant"):
            # FIX: We pass a formatted history string instead of an empty list
            res = qa({"question": prompt, "chat_history": st.session_state.chat_history})
            ans = res["answer"]
            
            # Update score
            if "Readiness Score:" in ans:
                try:
                    s = int(ans.split("Readiness Score:")[1].split("/")[0].strip())
                    st.session_state.score = s
                except: pass
            
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            # Save to history for the NEXT turn
            st.session_state.chat_history.append((prompt, ans))

elif choice == "📂 Document Library":
    st.header("Statute Downloads")
    st.write("Download Zimbabwean Acts for offline reading.")
