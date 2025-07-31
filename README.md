🧠✨ Multi-Brain Chat: Gemini & Local LLMs + RAG PDF QA

<div align="center"> <img src="https://img.shields.io/badge/streamlit-%F0%9F%A7%A1_lightning_-orange?style=for-the-badge" /> <img src="https://img.shields.io/badge/Google%20Gemini-%F0%9F%94%97_cloud-blue?style=for-the-badge" /> <img src="https://img.shields.io/badge/Local%20LLM-(CPU%20fast!)-green?style=for-the-badge" /> <img src="https://img.shields.io/badge/Supports-RAG_PDF_QA-%23FFC107?style=for-the-badge" /> <a href="https://github.com/Surajkecode"><img src="https://img.shields.io/badge/GitHub-Surajkecode-%231F2328?logo=github&logoColor=white&style=for-the-badge" /></a> <br />
  

📝 Project Overview
Welcome to Multi-Brain Chat — a powerful, modular Streamlit app that fuses Google Gemini cloud AI models, blazing-fast local quantized LLMs (Vicuna, Mistral, Llama2), and a Retrieval-Augmented Generation (RAG) system to answer questions from your PDFs.

Whether you want to generate and review code, chat with an empathetic AI, get expert medical advice, or deeply query your PDFs, Multi-Brain Chat is your all-in-one AI playground — with a clean, expressive UI and flexible backend selection.


## ✨ Features

- 🤖 **Multiple specialized chatbot modes:**
  - **Code Generator:** Prioritizes frontend (HTML/CSS/JS) code, extends to backend languages as needed, with explanations.
  - **Code Reviewer:** Provides detailed code analysis, improvements, and fixes.
  - **GF Bot:** Emotional, caring, affectionate AI companion.
  - **General Q&A:** Concise factual answers on diverse topics.
  - **Medical (Allopathy & Ayurveda):** Safe, conservative health advice tailored to user needs.

- 🔄 **Backend selection:**
  - **Google Gemini 1.5 Flash** & **2.5 Flash-Lite** (cloud API models)
  - **Local quantized LLMs:** Download and run on CPU with minimal RAM overhead.
  
- 📚 **Ask my PDFs (RAG):**
  - Upload multiple PDFs
  - Semantic vector indexing using sentence-transformers + FAISS
  - Ask questions with answers grounded in your documents, powered by Gemini
  
- 🎨 **Beautiful UI:**
  - Streamlit-native chat bubbles with avatars & emojis
  - Live streaming answers
  - File upload support for richer context
  - Download chat history for saving conversations


## 🚀 Getting Started

### Prerequisites

- Python 3.10 or above recommended
- API Key from Google Gemini (set in `.env`)

### Installation

```bash
git clone https://github.com/Surajkecode/multi-brain-chat.git
cd multi-brain-chat

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Setting Up the API Key

Create a file `.env` in the root directory:

```plaintext
GEMINI_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY_HERE
```

Replace `YOUR_GOOGLE_GEMINI_API_KEY_HERE` with your actual Gemini API key.

## 📦 Usage

Run the app locally:

```bash
streamlit run app.py
```

### Chatbots Section

- Select the preferred AI **backend**: Gemini 1.5 Flash, Gemini 2.5 Flash-Lite, or Local Quantized LLM.
- If local backend selected, choose a local model (Vicuna, Mistral, Llama 2) and download via sidebar button.
- Pick the desired **bot mode**.
- Upload files (*.txt, *.py, *.md*) for richer answers.
- Send messages and receive streaming replies inside chat bubbles.
- Download conversation logs at any time.

### Ask my PDFs Section

- Upload one or more PDF documents.
- The app extracts text, builds an in-memory semantic index.
- Ask questions about your documents.
- Get accurate, context-grounded answers using Gemini inference.

## 💾 Local Model Downloads

Models are automatically downloaded once when you select and click the button in the sidebar. Supported:

| Model                   | Quantization | Download Link (in app)                                                      |
|-------------------------|--------------|----------------------------------------------------------------------------|
| Vicuna 7B v1.5 Q4_K_S   | Q4_K_S       | https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF                       |
| Mistral 7B Instruct Q4_K_M | Q4_K_M    | https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF             |
| Llama 2 7B Chat Q4_K_M  | Q4_K_M       | https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF                      |

These models are optimized for CPU and ~8GB RAM systems.

## 🧰 Tech Stack

- Python 3.10+
- Streamlit (UI)
- Google GenerativeAI SDK
- llama-cpp-python (local LLM inference)
- PyPDF2 (PDF text extraction)
- sentence-transformers & FAISS (RAG index & search)
- requests (model download)
- python-dotenv (env management)
- numpy

## 🧩 Project Structure

```
multi-brain-chat/
├── app.py              # Main Streamlit app code
├── .env                # Your Gemini API key here (not committed)
├── requirements.txt    # List of python dependencies
├── models/             # Local quantized LLMs stored here after download
└── README.md           # This README file
```

## 💡 About This Project

**Multi-Brain Chat** stands out by combining multiple model backends, expert bot personalities, and document-based question answering in a single seamless UI. It’s:

- **Modular** — Add new bot modes or models easily.
- **User-friendly** — Intuitive interface, streamed replies, helpful context uploads.
- **Performance-aware** — Supports lightweight CPU LLMs with quantized weights.
- **Document-savvy** — Ask your own PDFs, research without manual lookup.

## 🌐 Connect with Me

| Icon | Contact | Link |
|-------|---------|-------|
| 📧 | Email | [surajborkute.tech@gmail.com](mailto:surajborkute.tech@gmail.com) |
| 💼 | LinkedIn | [Suraj Borkute](https://www.linkedin.com/in/suraj-borkute-87597b270) |
| 💻 | GitHub | [Surajkecode](https://github.com/Surajkecode) |
| 📱 | WhatsApp | [Message Now](https://wa.me/919518772281) or 📞 +91 9518772281 |

⭐ If you find this project helpful, please consider starring it on GitHub!

## 📢 License

MIT License © 2024 Suraj Borkute

Thank you for checking out **Multi-Brain Chat**!  
_I built this with ❤️ to help AI developers, researchers, and coders harness multiple powerful LLMs in one friendly app._  
Feel free to contribute, suggest features, or report issues!


**🎉 Happy chatting & coding!** 🎉

*Feel free to add screenshots, GIFs, or video demos to your GitHub repo to give users an instant visual sense of the UI's polish and power.*


