<div align="center">

# 🤖 DocuMind - Ask questions. Get answers.From your own documents

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3.0-green?logo=chainlink&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple?logo=meta&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-LLaMA3-orange?logo=ollama&logoColor=white)
![Offline](https://img.shields.io/badge/Mode-100%25%20Offline-red?logo=lock&logoColor=white)

**⚡ A fully offline  retrieval-based AI system for context-aware question answering from custom data.**

*Retrieve. Augment. Generate. Answer.*

</div>

---

## 📌 Table of Contents

- [What Is This Project?](#-what-is-this-project)
- [Why RAG?](#-why-rag)
- [How It Works](#-how-it-works)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [File Breakdown](#-file-breakdown)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Example Interaction](#-example-interaction)
- [Use Cases](#-use-cases)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)

---

## 🧠 What Is This Project?

This is a **fully offline, privacy-first AI chatbot** that reads and understands a custom text file (`data.txt`) and answers user questions about it — without sending any data to the cloud.

It combines three powerful technologies:

| Technology | Role |
|---|---|
| 🤗 HuggingFace Embeddings | Converts text into vector representations |
| ⚡ FAISS | Stores and searches vectors with lightning speed |
| 🦙 Ollama (LLaMA 3) | Generates intelligent, context-aware answers locally |

---

## 💡 Why RAG?

Traditional LLMs (Large Language Models) only know what they were trained on. They **cannot answer questions about your private documents** or custom data.

**RAG (Retrieval-Augmented Generation)** solves this by:

1. **Retrieving** the most relevant chunks from your document
2. **Injecting** those chunks as context into the LLM prompt
3. **Generating** a grounded, accurate answer — not a hallucination

```
Without RAG  →  LLM guesses from training data  →  Often wrong or outdated
With RAG     →  LLM reads YOUR document first   →  Accurate, grounded answers
```

---

## ⚙️ How It Works

Here is the complete pipeline, step by step:

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
│                                                                 │
│  data.txt                                                       │
│     │                                                           │
│     ▼                                                           │
│  [TextLoader]  ──→  Load raw text from file                     │
│     │                                                           │
│     ▼                                                           │
│  [CharacterTextSplitter]  ──→  Split into 500-char chunks       │
│     │                          with 50-char overlap             │
│     ▼                                                           │
│  [HuggingFaceEmbeddings]  ──→  Convert chunks to vectors        │
│     │                          (all-MiniLM-L6-v2 model)        │
│     ▼                                                           │
│  [FAISS Vector Store]  ──→  Store & index all vectors           │
│                                                                 │
│  ──── At Query Time ────────────────────────────────────────    │
│                                                                 │
│  User Question                                                  │
│     │                                                           │
│     ▼                                                           │
│  [FAISS Retriever]  ──→  Find top-k most relevant chunks        │
│     │                                                           │
│     ▼                                                           │
│  [Prompt Builder]  ──→  Format: Context + Question              │
│     │                                                           │
│     ▼                                                           │
│  [OllamaLLM / LLaMA 3]  ──→  Generate final answer             │
│     │                                                           │
│     ▼                                                           │
│  Bot Response printed to terminal                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Architecture

```
rag-chatbot/
│
├── data.txt           # 📄 Your custom knowledge base (edit this!)
├── embed_bot.py       # 🧠 Main chatbot script (RAG pipeline)
├── requirements.txt   # 📦 All Python dependencies
├── .gitignore         # 🚫 Files excluded from Git
└── README.md          # 📖 This file
```

---

## 📁 File Breakdown

### 📄 `data.txt` — The Knowledge Base
> **What it is:** The text document that the chatbot reads and answers questions from.

- Contains your custom knowledge (AI concepts in this example)
- You can **replace or extend** this with any text: product manuals, legal documents, medical notes, company policies, etc.
- The chatbot will **only answer based on this file** — no hallucinations from outside knowledge

**Why it's needed:** Without this file, the chatbot has no domain-specific knowledge to retrieve from. It is the single source of truth for the RAG system.

---

### 🧠 `embed_bot.py` — The Brain
> **What it is:** The main Python script that runs the entire RAG pipeline.

It handles all 10 stages of the system:

| Step | Code Component | Purpose |
|------|---------------|---------|
| 1 | `TextLoader` | Loads `data.txt` into memory |
| 2 | `CharacterTextSplitter` | Splits text into manageable 500-char chunks |
| 3 | `HuggingFaceEmbeddings` | Converts chunks into numerical vectors |
| 4 | `FAISS.from_documents` | Builds a searchable vector index |
| 5 | `db.as_retriever()` | Creates a retriever to find relevant chunks |
| 6 | `OllamaLLM` | Loads the local LLaMA 3 model |
| 7 | `format_docs()` | Combines retrieved chunks into a single context string |
| 8 | `build_prompt()` | Constructs the final prompt sent to the LLM |
| 9 | `rag_pipeline()` | Orchestrates the full query → retrieval → answer flow |
| 10 | `while True` loop | Interactive terminal chat interface |

**Why it's needed:** This is the core of the project. Every other file exists to support this script.

---

### 📦 `requirements.txt` — The Dependencies
> **What it is:** A list of all Python libraries needed to run the project.

```
langchain==0.3.0              # Core LangChain framework (chains, loaders, splitters)
langchain-community==0.3.0    # Community integrations (FAISS, TextLoader)
langchain-core==0.3.0         # Core abstractions (Runnables, Documents)
langchain-text-splitters==0.3.0  # Text chunking utilities
langchain-huggingface==0.0.3  # HuggingFace embeddings integration
langchain-ollama==0.1.0       # Ollama LLM integration
faiss-cpu                     # Facebook AI Similarity Search (vector DB)
sentence-transformers          # Powers the MiniLM embedding model
python-dotenv                 # Loads environment variables from .env file
```

**Why it's needed:** Python requires all third-party packages to be explicitly installed. This file lets anyone recreate your exact environment with a single command:
```bash
pip install -r requirements.txt
```

---

### 🚫 `.gitignore` — The Safety Net
> **What it is:** Tells Git which files to **never** commit to version control.

Typically includes:
```
__pycache__/       # Python bytecode cache — auto-generated, not needed
*.env              # Environment variables — may contain API keys/secrets
*.pyc              # Compiled Python files
faiss_index/       # Generated vector index — can be rebuilt anytime
```

**Why it's needed:** Prevents sensitive files (API keys, secrets) and auto-generated files (caches, build artifacts) from being accidentally uploaded to GitHub. Keeps your repository clean and secure.

---

## 🛠️ Tech Stack

| Component | Technology | Why This Choice |
|-----------|-----------|----------------|
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | Free, fast, runs locally, great semantic understanding |
| Vector Store | FAISS (Facebook AI) | Blazing fast similarity search, no server required |
| LLM | LLaMA 3 via Ollama | Free, runs 100% locally, no API key needed |
| Framework | LangChain | Clean abstractions for building LLM pipelines |
| Language | Python 3.10+ | Industry standard for AI/ML projects |

---

## ✅ Prerequisites

Before running this project, make sure you have:

- [ ] **Python 3.10+** installed
- [ ] **Ollama** installed on your system → [Download Ollama](https://ollama.com/download)
- [ ] **LLaMA 3 model** pulled via Ollama:
  ```bash
  ollama pull llama3
  ```
- [ ] Ollama server is **running** in the background:
  ```bash
  ollama serve
  ```

> ⚠️ **Important:** The chatbot will fail if Ollama is not running. Always start it before launching `embed_bot.py`.

---

## 🚀 Installation & Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### Step 2 — Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add Your Data
Edit `data.txt` and replace the content with your own knowledge base — any plain text works.

### Step 5 — Start Ollama
```bash
ollama serve
```

### Step 6 — Run the Chatbot
```bash
python embed_bot.py
```

---

## 💬 Usage

Once running, the terminal will display:

```
🤖 FREE Chatbot ready! Type 'exit' to stop.

You: 
```

Type any question related to your `data.txt` content and press **Enter**. Type `exit` to quit.

---

## 🧪 Example Interaction

Given the default `data.txt` about Artificial Intelligence:

```
You: What is Natural Language Processing?

Bot: Natural Language Processing (NLP) is a field of AI that focuses on enabling 
machines to understand, interpret, and generate human language. It is applied in 
chatbots, translation systems, sentiment analysis, and voice assistants like 
Siri or Alexa.

You: What are the challenges of AI?

Bot: AI faces several challenges including bias in data, lack of transparency, 
ethical concerns, and potential impact on jobs. This is why responsible AI 
development is emphasized to ensure fairness, accountability, and safety.

You: exit
```

---

## 🌍 Use Cases

This system can be adapted for many real-world applications:

- 📚 **Document Q&A** — Ask questions about legal, medical, or technical documents
- 🏢 **Enterprise Knowledge Base** — Internal FAQ bots with company-specific data
- 🎓 **Education** — Study assistants trained on textbooks or lecture notes
- 🏥 **Healthcare** — Query patient documents or clinical guidelines privately
- 🔒 **Privacy-First AI** — All data stays on your machine, nothing goes to the cloud

---

## ⚠️ Limitations

| Limitation | Description |
|------------|-------------|
| Context Window | Only retrieves top-k chunks; may miss info from very large documents |
| Single File | Currently supports one `data.txt`; multi-file support needs custom setup |
| No Memory | Each query is independent; the chatbot doesn't remember previous turns |
| Ollama Required | Must have Ollama running locally; not suitable for serverless deployment |
| Chunk Quality | Answer quality depends on how well the text chunks are split |

---

## 🔮 Future Improvements

- [ ] Support for **multiple files** (PDF, DOCX, CSV)
- [ ] Add **conversation memory** for multi-turn dialogue
- [ ] Build a **web UI** using Streamlit or Gradio
- [ ] Persist the **FAISS index** to disk to avoid re-embedding on every run
- [ ] Add **source citation** — show which chunk the answer came from
- [ ] Support **OpenAI / Groq API** as alternative LLM backends

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute with attribution.

<section>
<h2>👨‍💻 Author</h2>

<p><strong>Rohan</strong></p>

<p>
<a href="https://www.linkedin.com/in/rupjit-shil/" target="_blank">
<img src="https://img.shields.io/badge/LinkedIn-Rupjit%20Shil-blue?style=for-the-badge&logo=linkedin">
</a>
</p>

</section>

---

<div align="center">

Built with ❤️ using LangChain · HuggingFace · FAISS · Ollama

**Fully Offline · Completely Free · No API Keys Required**

</div>
