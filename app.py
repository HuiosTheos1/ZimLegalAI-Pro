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
    name = st.text_input("Enter your name to access the dashboard:")
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

if choice == "🏠 Home (Our Mission)":
    st.title("⚖️ Zim-Legal AI Hub")
    # THE CORRECTED MOTTO
    st.subheader("Motto: 'Justice Delayed is NOT Justice Denied.'")
    st.markdown("---")
    st.markdown("""
    ### 🛡️ Our Mission
    To empower the Zimbabwean citizen and business owner. We believe that regardless of how long the legal process 
    takes, the truth and your rights must be protected.
    
    ### 🚀 Sector Tools:
    - **Retailers:** Get your ZIMRA and Council checklists.
    - **Miners:** Understand pegging and EMA compliance.
    - **Transport:** Check VID and route permit requirements.
    - **AI Advisor:** Practice your defense and get a 'Readiness Score' before your court date.
    """)
    st.success(f"Welcome, {st.session_state.user_name}. Use the sidebar to navigate.")

elif choice == "🏢 Retail & Store Compliance":
    st.header("Retail & Storefront Operations")
    st.markdown("""
    - **Licensing:** Harare/Mutare/Local Council Shop Licenses.
    - **Taxes:** ZIMRA Income Tax and VAT compliance for 2026.
    - **Health:** Sanitary requirements for food outlets.
    """)

elif choice == "⛏️ Mining & Claims Law":
    st.header("Mining & Extractives Sector")
    st.markdown("""
    - **Claims:** Prospecting and pegging procedures.
    - **Environment:** Environmental Management Agency (EMA) EIA requirements.
    - **Gold:** Fidelity Gold Refinery sales protocols.
    """)

elif choice == "🚕 Transport & Taxi Regulations":
    st.header("Transport & Logistics")
    st.markdown("""
    - **Fitness:** VID testing and PSV requirements.
    - **Insurance:** Passenger and third-party liability.
    - **ZINARA:** Licensing and tollgate protocols.
    """)

elif choice == "⚖️ Legal AI Advisor (Pre-Trial)":
    st.header("Legal AI Interrogator")
    db = initialize_brain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Explain your legal challenge..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=st.secrets["GROQ_API_KEY"])
        
        TEMPLATE = f"""You are the 'Zim-Legal Advisor' for {st.session_state.user_name}.
        Challenge the user's story, cite Zimbabwean law where possible, and always end with:
        'Readiness Score: X/100'.
        
        CONTEXT: {{context}} | HISTORY: {{chat_history}}
        QUESTION: {{question}}"""
        
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=db.as_retriever(),
            combine_docs_chain_kwargs={"prompt": PromptTemplate(template=TEMPLATE, input_variables=["context", "chat_history", "question"])}
        )
        
        with st.chat_message("assistant"):
            # Fixed history handling
            res = qa({"question": prompt, "chat_history": st.session_state.chat_history})
            ans = res["answer"]
            
            if "Readiness Score:" in ans:
                try:
                    s = int(ans.split("Readiness Score:")[1].split("/")[0].strip())
                    st.session_state.score = s
                except: pass
            
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
            st.session_state.chat_history.append((prompt, ans))

elif choice == "📂 Document Library":
    st.header("Legal Document Library")
    st.write("Browse and download the statutes relevant to your business.")
