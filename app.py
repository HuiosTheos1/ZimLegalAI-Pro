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
        st.error("⚠️ No Law PDFs found. Please upload statutes to 'docs/' folder on GitHub.")
        st.stop()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

# --- 2. SETUP ---
st.set_page_config(page_title="Zim-Legal AI Hub", page_icon="⚖️", layout="wide")

if "user_name" not in st.session_state:
    st.title("⚖️ Zim-Legal AI")
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
    st.title(f"👤 {st.session_state.user_name}")
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
    st.subheader("Motto: 'Justice Delayed is NOT Justice Denied.'")
    st.markdown("---")
    st.markdown(f"""
    ### 🛡️ Welcome, {st.session_state.user_name}
    Our mission is to empower the Zimbabwean citizen and business owner with instant, 
    accurate legal information. We believe that legal knowledge should not be 
    hidden behind expensive fees.
    
    ### 🚀 Sector Specific Tools:
    - **Retail & Mining:** Checklists for 2026 compliance.
    - **Transport:** Licensing and VID requirements.
    - **Legal AI Advisor:** Test your legal defense against an adversarial AI.
    """)
    st.success("Please select a service from the sidebar to begin.")

elif choice == "🏢 Retail & Store Compliance":
    st.header("🏢 Retail & Store Compliance")
    st.write("Information on Shop Licenses, ZIMRA, and Council requirements.")

elif choice == "⛏️ Mining & Claims Law":
    st.header("⛏️ Mining & Claims Law")
    st.write("Guidance on Prospecting, Pegging, and EMA Compliance.")

elif choice == "🚕 Transport & Taxi Regulations":
    st.header("🚕 Transport & Taxi Regulations")
    st.write("Checklists for VID, Insurance, and Passenger permits.")

elif choice == "⚖️ Legal AI Advisor (Pre-Trial)":
    st.header("⚖️ Legal AI Advisor (Pre-Trial)")
    st.caption("Adversarial Mode: I will challenge your defense to prepare you for court.")
    
    db = initialize_brain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Explain your case or legal issue..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        try:
            llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=st.secrets["GROQ_API_KEY"])
            
            # Simplified template to prevent Groq BadRequest errors
            TEMPLATE = """You are the Zim-Legal Advisor. 
            Challenge the user's defense, cite Zimbabwean law, and always end with 'Readiness Score: X/100'.
            Context: {context}
            Chat History: {chat_history}
            User: {question}
            Advisor:"""
            
            qa = ConversationalRetrievalChain.from_llm(
                llm=llm, 
                retriever=db.as_retriever(),
                combine_docs_chain_kwargs={"prompt": PromptTemplate(template=TEMPLATE, input_variables=["context", "chat_history", "question"])}
            )
            
            with st.chat_message("assistant"):
                res = qa.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
                ans = res["answer"]
                
                if "Readiness Score:" in ans:
                    try:
                        s = int(ans.split("Readiness Score:")[1].split("/")[0].strip())
                        st.session_state.score = s
                    except: pass
                
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.session_state.chat_history.append((prompt, ans))
                
        except Exception as e:
            st.error(f"AI Connection Error: Please try again in a moment. (Error: {str(e)})")

elif choice == "📂 Document Library":
    st.header("📂 Document Library")
    st.write("Browse official Statutes and Acts.")
