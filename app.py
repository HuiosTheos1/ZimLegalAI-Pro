import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. THE ENGINE ---
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

# --- 2. CONFIG & NAVIGATION ---
st.set_page_config(page_title="Zim-Legal Business Hub", page_icon="🏢", layout="wide")

if "user_name" not in st.session_state:
    st.title("⚖️ Zim-Legal AI: Business & Court Strategy")
    st.info("Founder: Clyton Makate")
    name = st.text_input("Enter your name:")
    if st.button("Enter Dashboard"):
        if name:
            st.session_state.user_name = name
            st.session_state.score = 50
            st.rerun()
    st.stop()

with st.sidebar:
    st.title(f"Hi, {st.session_state.user_name}")
    choice = st.radio("Select Tool:", [
        "🏢 Retail & Store Compliance",
        "⛏️ Mining & Claims Law",
        "🚕 Transport & Taxi Regulations",
        "⚖️ Legal AI Advisor (Pre-Trial)",
        "📂 Document Library"
    ])
    st.divider()
    if choice == "⚖️ Legal AI Advisor (Pre-Trial)":
        st.metric("Case Readiness Score", f"{st.session_state.score}%")
        st.progress(st.session_state.score / 100)

# --- 3. BUSINESS GUIDES (RETAIL, MINING, TAXI) ---
if choice == "🏢 Retail & Store Compliance":
    st.header("Retail & Shop License Guide")
    st.markdown("1. **Local Authority License** 2. **ZIMRA Tax Clearance** 3. **Health Inspector Approval**")

elif choice == "⛏️ Mining & Claims Law":
    st.header("Mining Compliance & EIA")
    st.markdown("1. **Prospecting License** 2. **Mining Commissioner Registration** 3. **EMA Certification**")

elif choice == "🚕 Transport & Taxi Regulations":
    st.header("Public Service Vehicle (PSV) Rules")
    st.markdown("1. **VID Certificate of Fitness** 2. **Route Permit** 3. **Passenger Insurance**")

# --- 4. THE PRE-TRIAL / ADVISOR AI ---
elif choice == "⚖️ Legal AI Advisor (Pre-Trial)":
    st.header("Adversarial Legal Advisor")
    st.caption("I am programmed to challenge your story to prepare you for a Magistrate or Labour Officer.")
    db = initialize_brain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "I am ready. State your case, a dispute, or a compliance issue you are facing."}]
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Explain your situation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=st.secrets["GROQ_API_KEY"])
        
        # PRE-TRIAL PROMPT LOGIC
        TEMPLATE = f"""You are the 'Zim-Legal Pre-Trial Teacher' for {st.session_state.user_name}.
        1. CHALLENGE the user's story like a tough Prosecutor.
        2. WARN if they say something legally damaging.
        3. CITE Zimbabwean Statutes where possible.
        4. End EVERY reply with 'Readiness Score: X/100'.
        
        CONTEXT: {{context}} | HISTORY: {{chat_history}}
        QUESTION: {{question}}
        Response:"""
        
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=db.as_retriever(),
            combine_docs_chain_kwargs={"prompt": PromptTemplate(template=TEMPLATE, input_variables=["context", "chat_history", "question"])}
        )
        
        with st.chat_message("assistant"):
            res = qa({"question": prompt, "chat_history": []})
            ans = res["answer"]
            
            # Score Update
            if "Readiness Score:" in ans:
                try:
                    s = int(ans.split("Readiness Score:")[1].split("/")[0].strip())
                    st.session_state.score = s
                except: pass
            
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

elif choice == "📂 Document Library":
    st.header("Statute Downloads")
    st.write("Access the full text of the Law.")
