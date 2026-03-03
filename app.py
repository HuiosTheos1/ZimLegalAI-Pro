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
    # Load all PDFs from the docs folder
    loader = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        st.error("⚠️ No Law PDFs found. Please upload statutes to 'docs/' folder on GitHub.")
        st.stop()
    
    # Split text into chunks for the AI to process
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create the vector database using HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

# --- 2. CONFIG & SESSION SETUP ---
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
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

# --- 4. THE PAGES ---

# --- HOME PAGE ---
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

# --- SECTOR GUIDES ---
elif choice == "🏢 Retail & Store Compliance":
    st.header("🏢 Retail & Store Compliance")
    st.markdown("""
    ### Business Essentials:
    - **Shop License:** Required from Local Council (e.g., Harare City Council).
    - **ZIMRA:** Register for BP Number and Tax Clearance (ITF263).
    - **NSSA:** Register as an employer and contribute for workers.
    - **Health Cert:** Mandatory for any premises handling food.
    """)

elif choice == "⛏️ Mining & Claims Law":
    st.header("⛏️ Mining & Claims Law")
    st.markdown("""
    ### Compliance Checklist:
    - **Prospecting License:** Ministry of Mines & Mining Development.
    - **Pegging:** Must be done by an approved Prospector.
    - **EMA:** Environmental Impact Assessment (EIA) is required before operations.
    - **Fidelity:** All gold production must be declared and sold via Fidelity Gold Refinery.
    """)

elif choice == "🚕 Transport & Taxi Regulations":
    st.header("🚕 Transport & Taxi Regulations")
    st.markdown("""
    ### Public Service Vehicle (PSV) Requirements:
    - **VID:** Current Certificate of Fitness.
    - **ZINARA:** Valid vehicle licensing and radio license.
    - **Insurance:** Full Passenger Liability Insurance.
    - **Permits:** Route Authority and Operator's License.
    """)

# --- AI ADVISOR (THE CORE AI) ---
elif choice == "⚖️ Legal AI Advisor (Pre-Trial)":
    st.header("⚖️ Legal AI Advisor (Pre-Trial)")
    st.caption("Adversarial Mode: I will challenge your defense using Zimbabwean Law.")
    
    db = initialize_brain()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display Chat History
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("State your legal challenge (e.g., 'I am accused of...')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        try:
            # UPDATED: Using Llama 3.3 for 2026 compatibility
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"])
            
            # Robust Prompt Template
            TEMPLATE = """You are the Zim-Legal Advisor for {user}.
            Challenge the user's defense, cite relevant Zimbabwean statutes from the provided context, 
            and always end your response with 'Readiness Score: X/100'.
            
            Context: {context}
            Chat History: {chat_history}
            User Question: {question}
            Advisor Response:"""
            
            qa = ConversationalRetrievalChain.from_llm(
                llm=llm, 
                retriever=db.as_retriever(),
                combine_docs_chain_kwargs={"prompt": PromptTemplate(template=TEMPLATE, input_variables=["user", "context", "chat_history", "question"])}
            )
            
            with st.chat_message("assistant"):
                # Use invoke (the latest LangChain method)
                res = qa.invoke({
                    "user": st.session_state.user_name,
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                ans = res["answer"]
                
                # Update Score Gauge
                if "Readiness Score:" in ans:
                    try:
                        s_str = ans.split("Readiness Score:")[1].split("/")[0].strip()
                        st.session_state.score = int(s_str)
                    except: pass
                
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.session_state.chat_history.append((prompt, ans))
                
        except Exception as e:
            st.error(f"AI Connection Error: {str(e)}")

# --- DOCUMENT LIBRARY ---
elif choice == "📂 Document Library":
    st.header("📂 Document Library")
    st.write("Below are the available Zimbabwean Acts in the database:")
    if os.path.exists('./docs/'):
