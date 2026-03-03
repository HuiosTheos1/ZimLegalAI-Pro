import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from engine import initialize_brain

# --- 1. PERSONA ENGINE (Clyton's Vision) ---
def get_system_prompt(name, role, lang):
    base_style = f"You are the 'Zim-Legal Pre-Trial Teacher' coaching {name}."
    
    # Adapt persona based on role
    if role == "Accused Person":
        instructions = "Focus on Constitutional rights (Section 50) and procedural defenses."
    elif role == "Employer/Company":
        instructions = "Focus on Labour Act compliance and avoiding unfair dismissal claims."
    elif role == "Lawyer":
        instructions = "Use high-level legal jargon and cite specific Section/Chapter numbers."
    else: # Student/Practice
        instructions = "Be educational and explain the logic of the statutes."

    return PromptTemplate(
        template=f"""{base_style} {instructions}
        LANGUAGE: Respond primarily in {lang}.
        CONTEXT FROM ZIMBABWEAN LAW: {{context}}
        HISTORY: {{chat_history}}
        USER INPUT: {{question}}

        RULES:
        1. ADVERSARIAL: Roleplay as a tough Magistrate or Prosecutor to test their story.
        2. WARNING: If they say something legally damaging, say "⚠️ STOP. {name}, do not say that because..."
        3. MITIGATION: Always ask if this is their first offense.
        4. SCORE: End every reply with 'Readiness Score: X/100'.
        """,
        input_variables=["context", "chat_history", "question"]
    )

# --- 2. THE GATEKEEPER (Intake Form) ---
st.set_page_config(page_title="Zim-Legal AI", page_icon="⚖️", layout="wide")

if "user_name" not in st.session_state:
    st.title("⚖️ Zim-Legal AI: Access to Justice")
    st.write(f"### Welcome to the Pre-Trial Simulator")
    with st.form("intake"):
        name = st.text_input("Full Name")
        lang = st.selectbox("Preferred Language", ["English", "Shona", "Ndebele", "Mixed"])
        role = st.selectbox("I am an:", ["Accused Person", "Employee", "Employer/Company", "Lawyer", "Student"])
        if st.form_submit_button("Begin Legal Strategy"):
            if name:
                st.session_state.user_name = name
                st.session_state.user_role = role
                st.session_state.lang = lang
                st.session_state.score = 50
                st.rerun()
            else: st.error("Please enter your name.")
    st.stop()

# --- 3. THE EMERGENCY SHIELD ---
if st.sidebar.button("🚨 EMERGENCY: I've been Arrested", type="primary"):
    st.session_state.emergency = True

if st.session_state.get('emergency'):
    st.error("### 🔴 CONSTITUTIONAL EMERGENCY PROTOCOL (Section 50)")
    st.markdown("""
    1. **SAY NOTHING:** Tell the police: "I wish to remain silent until my lawyer is present."
    2. **48 HOURS:** You must be brought to court within 48 hours or you must be released.
    3. **NO ASSAULT:** If you are beaten, tell the Magistrate immediately.
    4. **ONE CALL:** You have the right to call your family or lawyer immediately.
    """)
    if st.button("Return to Strategy Session"):
        st.session_state.emergency = False
        st.rerun()
    st.stop()

# --- 4. MAIN APP LOGIC ---
st.title(f"⚖️ {st.session_state.user_name}'s Pre-Trial Session")

# Load Brain from engine.py
if 'db' not in st.session_state:
    with st.spinner("Analyzing Zimbabwean Statutes..."):
        st.session_state.db = initialize_brain()

# Sidebar Stats
with st.sidebar:
    st.header("📊 Case Readiness")
    st.metric("Probability of Success", f"{st.session_state.score}%")
    st.progress(st.session_state.score / 100)
    st.divider()
    st.write(f"**User:** {st.session_state.user_name}")
    st.write(f"**Role:** {st.session_state.user_role}")
    if st.button("📥 Download Court Prep File"):
        summary = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Save Strategy", summary, file_name="ZimLegal_Strategy.txt")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Hello {st.session_state.user_name}. I am your Teacher. Tell me about your case."}]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# --- 5. THE INTERROGATION LOOP ---
if prompt := st.chat_input("Explain your situation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=st.secrets["GROQ_API_KEY"])
    PROMPT = get_system_prompt(st.session_state.user_name, st.session_state.user_role, st.session_state.lang)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=st.session_state.db.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    with st.chat_message("assistant"):
        res = qa({"question": prompt, "chat_history": st.session_state.chat_history})
        ans = res["answer"]
        
        # Update Score from AI text
        if "Readiness Score:" in ans:
            try: st.session_state.score = int(ans.split("Readiness Score:")[1].split("/")[0].strip())
            except: pass
            
        st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.session_state.chat_history.append((prompt, ans))
