import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


st.set_page_config(
    page_title="Resume Genie",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────
# CUSTOM CSS — Deep Indigo + Teal Theme
# ───────────────────────────────────────────────
st.markdown("""
<style>

/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --navy-deep:   #1a1a2e;
    --navy-mid:    #16213e;
    --navy-strong: #0f3460;
    --teal:        #1d9e75;
    --teal-light:  #9fe1cb;
    --teal-bg:     #e1f5ee;
    --text-light:  #e8eaf0;
    --text-muted:  #8b9ab5;
    --card-bg:     #1e2a45;
    --card-border: #2a3a5c;
    --input-bg:    #1a2540;
    --white:       #ffffff;
    --radius:      12px;
    --radius-sm:   8px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    min-height: 100vh;
}

/* ── Header ── */
[data-testid="stHeader"] {
    background-color: rgba(26, 26, 46, 0.8) !important;
    backdrop-filter: blur(10px) !important;
    border-bottom: 1px solid rgba(29, 158, 117, 0.15) !important;
}

[data-testid="stHeader"] * {
    color: var(--text-light) !important;
    fill: var(--text-light) !important;
}

/* ── Main content area ── */
.main .block-container {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(29,158,117,0.15);
    padding: 2rem 2.5rem !important;
    margin-top: 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f3460 0%, #16213e 100%) !important;
    border-right: 1px solid rgba(29,158,117,0.3) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-light) !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: var(--text-light) !important;
    font-size: 14px !important;
    padding: 8px 12px;
    border-radius: var(--radius-sm);
    transition: background 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(29,158,117,0.15) !important;
}

[data-testid="stSidebar"] .stMarkdown strong {
    color: var(--teal-light) !important;
    font-size: 20px !important;
    letter-spacing: 0.5px;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--teal-light) !important;
}

/* ── Page title (h1) ── */
h1 {
    color: var(--white) !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem !important;
}

/* ── Section headers (h2, h3) ── */
h2, h3 {
    color: var(--teal-light) !important;
    font-weight: 600 !important;
}

/* ── Body text & markdown ── */
p, .stMarkdown p, label, .stText, li {
    color: var(--text-light) !important;
}

/* ── Muted description under title ── */
.stMarkdown p strong {
    color: var(--teal-light) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d9e75, #0f6e56) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.3px;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(29,158,117,0.35) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #23b885, #1d9e75) !important;
    box-shadow: 0 6px 20px rgba(29,158,117,0.5) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(29,158,117,0.15) !important;
    color: var(--teal-light) !important;
    border: 1px solid rgba(29,158,117,0.4) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(29,158,117,0.25) !important;
    border-color: var(--teal) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: var(--input-bg) !important;
    color: var(--text-light) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 13.5px !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s !important;
}

.stTextArea textarea:disabled {
    color: var(--text-light) !important;
    -webkit-text-fill-color: var(--text-light) !important;
    opacity: 0.85 !important;
}

.stTextArea textarea:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px rgba(29,158,117,0.2) !important;
}

/* ── Text input ── */
.stTextInput input {
    background: var(--input-bg) !important;
    color: var(--text-light) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius-sm) !important;
}

.stTextInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px rgba(29,158,117,0.2) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--input-bg) !important;
    border: 2px dashed rgba(29,158,117,0.4) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--teal) !important;
}

[data-testid="stFileUploader"] * {
    color: var(--text-muted) !important;
}

[data-testid="stFileUploader"] small {
    color: var(--text-muted) !important;
}

/* ── Success / warning / info boxes ── */
.stSuccess {
    background: rgba(29,158,117,0.12) !important;
    border: 1px solid rgba(29,158,117,0.4) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--teal-light) !important;
}

.stWarning {
    background: rgba(239,159,39,0.1) !important;
    border: 1px solid rgba(239,159,39,0.35) !important;
    border-radius: var(--radius-sm) !important;
}

.stInfo {
    background: rgba(55,138,221,0.1) !important;
    border: 1px solid rgba(55,138,221,0.35) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--teal) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--card-bg) !important;
    color: var(--text-light) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--card-border) !important;
    font-weight: 500 !important;
}

.streamlit-expanderContent {
    background: var(--input-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-top: none !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.75rem !important;
}

[data-testid="stChatMessage"] p {
    color: var(--text-light) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--input-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius) !important;
}

[data-testid="stChatInput"] textarea {
    color: #1a1a2e !important;
}

/* ── Columns / card panels ── */
[data-testid="column"] {
    background: rgba(255,255,255,0.02);
    border-radius: var(--radius);
    padding: 0.5rem;
}

/* ── Selectbox / radio ── */
.stRadio > div {
    gap: 6px !important;
}

.stRadio div[data-testid="stMarkdownContainer"] p {
    color: var(--text-light) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--navy-mid);
}
::-webkit-scrollbar-thumb {
    background: var(--teal);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--teal-light);
}

/* ── Divider ── */
hr {
    border-color: rgba(29,158,117,0.2) !important;
}

/* ── Subheader labels ── */
.stSubheader {
    color: var(--teal-light) !important;
    font-weight: 600 !important;
}

/* ── Markdown output (AI response) ── */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--teal-light) !important;
}

.stMarkdown code {
    background: rgba(29,158,117,0.15) !important;
    color: var(--teal-light) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

.stMarkdown blockquote {
    border-left: 3px solid var(--teal) !important;
    padding-left: 1rem !important;
    color: var(--text-muted) !important;
}

/* ── Tables ── */
.stMarkdown table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1rem 0 !important;
    color: var(--text-light) !important;
}

.stMarkdown th {
    background-color: rgba(29, 158, 117, 0.15) !important;
    color: var(--teal-light) !important;
    border: 1px solid rgba(29, 158, 117, 0.25) !important;
    padding: 8px 12px !important;
    text-align: left !important;
    font-weight: 600 !important;
}

.stMarkdown td {
    border: 1px solid rgba(29, 158, 117, 0.15) !important;
    padding: 8px 12px !important;
    color: var(--text-light) !important;
}

/* ── Tooltip ── */
.stTooltipIcon {
    color: var(--teal) !important;
}

</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# SIDEBAR BRANDING
# ───────────────────────────────────────────────
st.sidebar.markdown("**✨ Resume Genie**")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    _api_key_missing = True
else:
    _api_key_missing = False

@st.cache_resource(show_spinner="🔄 Initializing AI model...")
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY, temperature=0.2, max_tokens=2000)

llm = get_llm()

# ───────────────────────────────────────────────
# SHARED PDF LOADER
# ───────────────────────────────────────────────
@st.cache_data
def extract_resume_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text = "\n\n".join(doc.page_content for doc in docs)
        return text
    finally:
        os.unlink(tmp_path)

# ───────────────────────────────────────────────
# PROMPTS
# ───────────────────────────────────────────────
COVER_LETTER_PROMPT = PromptTemplate.from_template("""
Write a professional cover letter (300–450 words) for this job. Match resume to JD exactly. Standard format.
Job Description: {job_description}
Resume: {resume_text}
Do not invent facts.
""")

RESUME_SCORER_PROMPT = """You are an expert resume scorer. Analyze match to JD. EXACT structure:
**Score**: X/100
**Overall Match**: X%
Keywords matched: • ...
Missing keywords: • ...
Readability Score: X/100
ATS Compatibility Score: X/100
2-liner summary: ...
Skill gap analysis: • ...
Overall improvement suggestions: • ...
Industry specific feedback: • ...
Job: {job_description}
Resume: {context}
Be honest, use rubrics."""

RESUME_CHECKER_PROMPT = PromptTemplate.from_template("""
Score resume standalone (clarity, format, ATS, skills): EXACT structure:
1. **Score**: X/100
2. **Strengths**: • ...
3. **Weaknesses**: • ...
4. **Skills Mentioned**: • ...
5. **Recommended Skills**: • ...
6. **Next Career Steps**: • ...
Resume: {context}
""")

# ───────────────────────────────────────────────
# MAIN UI
# ───────────────────────────────────────────────
st.title("🚀 Resume Genie")
st.markdown("""
**Powered by GROQ AI** • Your all-in-one solution for job applications  
**AI Tools** to craft winning resumes, cover letters & career strategies 💼✨
""")

# ─── SIDEBAR: Tool Selector ───
st.sidebar.title("🛠️ Select Tool")
tool = st.sidebar.radio("Choose a service:", [
    "✉️ Cover Letter Generator",
    "📊 Resume-JD Matcher",
    "🔍 Resume Checker",
    "💬 Career Coach Chat"
], index=0, horizontal=False)

# Shared inputs
if tool in ["✉️ Cover Letter Generator", "📊 Resume-JD Matcher"]:
    st.sidebar.subheader("📤 Inputs")
    job_desc = st.sidebar.text_area("Job Description", height=200, key="jd_shared")
    resume_file = st.sidebar.file_uploader("Resume PDF", type="pdf", key="resume_shared")

# ───────────────────────────────────────────────
# TOOL 1: COVER LETTER
# ───────────────────────────────────────────────
if tool == "✉️ Cover Letter Generator":
    st.header("✉️ AI Cover Letter Generator")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Job Description")
        job_description = st.text_area("Paste JD", value=job_desc or "", height=350, key="jd_cl")

    with col2:
        st.subheader("📄 Your Resume")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="cl_resume")
        if uploaded_file:
            if st.button("🔥 Generate Cover Letter", type="primary"):
                with st.spinner("Extracting → Generating..."):
                    resume_text = extract_resume_text(uploaded_file)
                    chain = COVER_LETTER_PROMPT | llm
                    full_response = ""
                    resp_container = st.empty()
                    for chunk in chain.stream({"job_description": job_description, "resume_text": resume_text}):
                        content = chunk.content if hasattr(chunk, "content") else str(chunk)
                        full_response += content
                        resp_container.markdown(full_response + "▌")
                    resp_container.markdown(full_response)
                    st.download_button("💾 Download .md", full_response, "cover_letter.md")

# ───────────────────────────────────────────────
# TOOL 2: RESUME SCORER/MATCHER
# ───────────────────────────────────────────────
elif tool == "📊 Resume-JD Matcher":
    st.header("📊 Resume vs Job Description Matcher")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Job Description")
        job_description = st.text_area("Paste full JD", value=job_desc or "", height=350, key="jd_scorer")

    with col2:
        st.subheader("📄 Resume")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="scorer_resume")
        if uploaded_file:
            st.success("✅ Resume loaded")
            if st.button("📈 Score Match", type="primary"):
                with st.spinner("Analyzing match... (30-60s)"):
                    context = extract_resume_text(uploaded_file)
                    prompt = RESUME_SCORER_PROMPT.format(job_description=job_description, context=context)
                    response = llm.invoke(prompt)
                    st.markdown("### 📊 **Analysis Result**")
                    st.markdown(response.content)

# ───────────────────────────────────────────────
# TOOL 3: RESUME CHECKER
# ───────────────────────────────────────────────
elif tool == "🔍 Resume Checker":
    st.header("🔍 Standalone Resume Evaluator")
    uploaded_file = st.file_uploader("Upload resume PDF", type="pdf", key="checker_resume")

    if uploaded_file and st.button("Evaluate Resume", type="primary"):
        with st.spinner("Evaluating..."):
            context = extract_resume_text(uploaded_file)
            chain = RESUME_CHECKER_PROMPT | llm
            response = chain.invoke({"context": context})
            st.markdown("### 📋 **Detailed Evaluation**")
            st.markdown(response.content)

# ───────────────────────────────────────────────
# TOOL 4: CAREER COACH CHAT
# ───────────────────────────────────────────────
elif tool == "💬 Career Coach Chat":
    st.header("💬 Career Coach Chatbot")

    if "resume_context" not in st.session_state:
        st.session_state.resume_context = None
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Upload resume first", type="pdf", key="chat_resume")
    if uploaded_file and st.session_state.resume_context is None:
        context = extract_resume_text(uploaded_file)
        st.session_state.resume_context = context
        st.rerun()

    if not st.session_state.resume_context:
        st.warning("👆 Upload your resume to start chatting!")
        st.stop()

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📄 Your Resume")
        with st.expander("View full text", expanded=True):
            st.text_area("", st.session_state.resume_context, height=500, disabled=True)

    with right_col:
        st.subheader("🤖 Career Coach")
        system_msg = SystemMessage(content=f"""You are a career coach. Use this resume: {st.session_state.resume_context}""")

        for msg in st.session_state.chat_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

        if prompt := st.chat_input("Ask about career, resume, interviews..."):
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            with st.chat_message("assistant"):
                messages = [system_msg] + st.session_state.chat_history
                resp_container = st.empty()
                full_resp = ""
                for chunk in llm.stream(messages):
                    full_resp += chunk.content
                    resp_container.markdown(full_resp + "▌")
                resp_container.markdown(full_resp)
            st.session_state.chat_history.append(AIMessage(content=full_resp))
            st.rerun()