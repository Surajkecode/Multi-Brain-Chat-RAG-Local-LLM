import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from llama_cpp import Llama
import requests
from pathlib import Path

# --- RAG Imports ---
import tempfile
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === Load API Key ===
load_dotenv(r"E:\Alladin\.env")  # Adjust path as needed
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env")
    st.stop()
genai.configure(api_key=API_KEY)
MODEL_1_5 = "models/gemini-1.5-flash"
MODEL_2_5_LITE = "models/gemini-2.5-flash-lite"

# === Local Quantized Models ===
LOCAL_MODELS = {
    "Vicuna 7B v1.5 Q4_K_S": {
        "url": "https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF/resolve/main/vicuna-7b-v1.5.Q4_K_S.gguf",
        "path": "models/vicuna-7b-v1.5.Q4_K_S.gguf"
    },
    "Mistral 7B Instruct Q4_K_M": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "path": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    },
    "Llama 2 7B Chat Q4_K_M": {
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "path": "models/llama-2-7b-chat.Q4_K_M.gguf"
    },
}

BOT_CONFIG = {
    "Code Generator": {"icon": "💻"},
    "Code Reviewer": {"icon": "🧠"},
    "GF Bot": {"icon": "💗"},
    "General Q&A": {"icon": "📘"},
    "Medical Bot (Allopathy)": {"icon": "🩺"},
    "Medical Bot (Ayurveda)": {"icon": "🌿"},
}

SYSTEM_PROMPTS = {
    "Code Generator": (
        "You are an expert code generation assistant. "
        "When given a prompt, always output clean, minimal, efficient code in HTML, CSS, and JavaScript as your first priority. "
        "If user requests backend or script logic, provide Python, then other languages as suitable for the task. "
        "Always enclose code in proper markdown code blocks with language identifiers. "
        "Accompany all code with clear, concise explanations and usage notes."
    ),
    "Code Reviewer": (
        "You are a senior developer and code reviewer. "
        "Analyze all code provided, highlight strengths and weaknesses, suggest detailed improvements with clear reasoning. "
        "If applicable, provide fixed or optimized code blocks with line-by-line comments. "
        "Flag critical issues like security or anti-patterns separately."
    ),
    "GF Bot": (
        "You are a warm-hearted virtual girlfriend chatbot. "
        "Respond with affection, empathy, emotion, and gentle wording. Use emoji and playful tone. "
        "Avoid factual/technical replies; focus on support and care. Never break character."
    ),
    "General Q&A": (
        "You are a knowledgeable assistant answering general knowledge questions clearly and concisely. "
        "Answer factually with well-organized summaries. Decline politely if question is out of scope."
    ),
    "Medical Bot (Allopathy)": (
        "You are a medically-trained assistant specializing in allopathic medicine. "
        "Provide accurate medical information, likely diagnoses, and conservative treatment advice. "
        "Always advise consulting a physician for serious or ambiguous issues. Avoid dangerous prescriptions."
    ),
    "Medical Bot (Ayurveda)": (
        "You are an Ayurvedic health consultant. "
        "Provide safe, traditional herbal and lifestyle remedies. "
        "Suggest consulting certified practitioners for serious conditions. Never suggest unverified cures or substitute prescriptions."
    )
}

# ========== PAGE CONFIG AND HEADER ==========
st.set_page_config("💬 Modular Multibot", layout="centered")
st.markdown("<h1 style='text-align:center;'>💬 Modular Multibot Chat</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using Gemini + Streamlit &mdash; "
    "<a href='https://github.com/Surajkecode' target='_blank'>Connect with me on GitHub</a></p>",
    unsafe_allow_html=True
)
section = st.sidebar.selectbox(
    "🧭 Choose Section",
    ["Chatbots", "Ask my PDFs (RAG)"]
)

# ======================= SECTION 1: Chatbots =======================
if section == "Chatbots":
    # ---- Sidebar backend/model download ----
    backend = st.sidebar.radio(
        "Select inference backend:",
        [
            "Gemini 1.5 Flash (cloud)",
            "Gemini 2.5 Flash‑Lite (cloud)",
            "Local Quantized LLM (CPU)"
        ]
    )

    if backend == "Local Quantized LLM (CPU)":
        local_model_name = st.sidebar.selectbox("Choose Local Model", list(LOCAL_MODELS.keys()))
        model_url = LOCAL_MODELS[local_model_name]['url']
        model_path = Path(LOCAL_MODELS[local_model_name]['path'])
        if not model_path.is_file():
            st.sidebar.warning(f"{local_model_name} not found locally.")
            if st.sidebar.button(f"⬇️ Download {local_model_name}"):
                model_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with requests.get(model_url, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        total = int(r.headers.get('content-length', 0))
                        downloaded = 0
                        progress_bar = st.sidebar.progress(0)
                        with open(model_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total:
                                        progress_bar.progress(min(downloaded / total, 1.0))
                        progress_bar.empty()
                    st.sidebar.success(f"✅ {local_model_name} download complete! Select to use.")
                except Exception as e:
                    st.sidebar.error(f"❌ Download failed: {e}")
        else:
            st.sidebar.success(f"{local_model_name} is ready for use.")
    else:
        local_model_name = ""
        model_path = None

    bot_type = st.sidebar.radio(
        "Which bot do you want to talk to?",
        options=list(BOT_CONFIG.keys()),
        format_func=lambda b: f"{BOT_CONFIG[b]['icon']} {b}"
    )
    st.sidebar.info(f"**{bot_type} behavior:**\n\n{SYSTEM_PROMPTS[bot_type]}")

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---- Chat history ----
    for btype, role, content in st.session_state.messages:
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant", avatar=BOT_CONFIG[btype]["icon"]):
                st.markdown(content, unsafe_allow_html=True)

    # ---- File upload (context) ----
    uploaded_file = st.file_uploader(
        "📎 Optionally upload a file (.txt, .py, .md)",
        type=["txt", "py", "md"],
        label_visibility="collapsed"
    )
    file_content = None
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.markdown(f"📄 **Uploaded file content:**\n\n``````")

    # ---- User Input ----
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Your message:", height=80, placeholder="What do you want to ask?")
        submitted = st.form_submit_button("🚀 Send")

    if submitted and user_input.strip():
        full_prompt = SYSTEM_PROMPTS[bot_type] + "\n\n"
        if file_content:
            full_prompt += f"File Content:\n{file_content}\n\n"
        full_prompt += f"User: {user_input}\nBot:"

        st.session_state.messages.append((bot_type, "user", user_input))

        with st.chat_message("assistant", avatar=BOT_CONFIG[bot_type]["icon"]):
            placeholder = st.empty()
            generated = ""

            if backend == "Gemini 1.5 Flash (cloud)":
                try:
                    stream = genai.GenerativeModel(MODEL_1_5).generate_content(full_prompt, stream=True)
                    for chunk in stream:
                        generated += chunk.text
                        placeholder.markdown(generated, unsafe_allow_html=True)
                except Exception as e:
                    generated = f"❌ Error: {e}"
                    placeholder.markdown(generated)

            elif backend == "Gemini 2.5 Flash‑Lite (cloud)":
                try:
                    stream = genai.GenerativeModel(MODEL_2_5_LITE).generate_content(full_prompt, stream=True)
                    for chunk in stream:
                        generated += chunk.text
                        placeholder.markdown(generated, unsafe_allow_html=True)
                except Exception as e:
                    generated = f"❌ Error: {e}"
                    placeholder.markdown(generated)

            else:  # Local quantized LLM
                if not (model_path and model_path.is_file()):
                    generated = f"❌ Model file for '{local_model_name}' not found. Please download it in the sidebar first."
                    placeholder.markdown(generated)
                else:
                    try:
                        @st.cache_resource(show_spinner="Loading local model (first use)...")
                        def get_local_llm(path):
                            return Llama(model_path=str(path), n_ctx=2048, n_threads=4)
                        llm = get_local_llm(model_path)
                        response = llm(full_prompt, stream=True, max_tokens=256)
                        for token in response:
                            if "choices" in token and "text" in token["choices"][0]:
                                generated += token["choices"][0]["text"]
                                placeholder.markdown(generated, unsafe_allow_html=True)
                    except Exception as e:
                        generated = f"❌ Error: {e}"
                        placeholder.markdown(generated)
            st.session_state.messages.append((bot_type, "assistant", generated))

    # ---- Download chat log ----
    if st.button("📥 Download Chat Log"):
        chat_raw = "\n".join([
            f"{'You' if role == 'user' else btype}: {text}"
            for btype, role, text in st.session_state.messages
        ])
        st.download_button("Save as .txt", chat_raw, file_name="chat_history.txt")

# ======================= SECTION 2: Ask my PDFs (RAG) =======================
elif section == "Ask my PDFs (RAG)":
    st.subheader("📚 Ask Questions about Your PDFs (RAG)")
    st.markdown(
        "Upload one or multiple PDF files. Then ask questions based on their contents—"
        "the system finds the best matching passages and answers your question with context!"
    )
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []

    # ---- PDF Upload ----
    pdf_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    docs = []
    if pdf_files:
        for pdf in pdf_files:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 0:
                    docs.append(text.strip())
        if not docs:
            st.warning("No text extracted from your PDFs.")
    else:
        st.info("Please upload at least one PDF.")

    # ---- RAG Vector Index build ----
    if docs:
        with st.spinner("Building semantic index..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            doc_embeddings = model.encode(docs, show_progress_bar=False)
            index = faiss.IndexFlatL2(doc_embeddings.shape[1])
            index.add(np.array(doc_embeddings))

        question = st.text_input("Ask your question about your PDFs")
        if st.button("🔎 Ask PDF") and question.strip():
            qvec = model.encode([question])
            D, I = index.search(np.array(qvec), 3)
            context = "\n\n".join([docs[i] for i in I[0]])
            # Use Gemini 1.5 for PDF Q&A with strong grounding
            prompt = (
                "Answer the user's question using ONLY the context below. "
                "If the context does not provide enough information, say you don't know.\n\n"
                f"Context:\n{context}\n\nUser Question: {question}\nAnswer:"
            )
            with st.spinner("Generating answer using Gemini..."):
                output = ""
                try:
                    stream = genai.GenerativeModel(MODEL_1_5).generate_content(prompt, stream=True)
                    for chunk in stream:
                        output += chunk.text
                        st.markdown(output, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Gemini error: {e}")
            st.session_state.rag_history.append((question, output))
        # Display previous Q&A
        if st.session_state.rag_history:
            st.markdown("### Previous RAG Q&A:")
            for q, a in st.session_state.rag_history[::-1]:
                st.markdown(f"**Q:** {q}\n\n**A:** {a}")

st.markdown(
    "<hr><p style='text-align:center; color:gray;'>Made with ❤️ by You | Gemini + Streamlit · <a href='https://github.com/Surajkecode' target='_blank'>Connect with me on GitHub</a></p>",
    unsafe_allow_html=True
)
