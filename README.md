<div align="center">

<img src="https://github.com/user-attachments/assets/f6f6c9de-825e-49da-9363-e107520c68e1" alt="NIT Rourkela AI Agent Assistant banner" width="100%"/>

<br/>

<h1>🤖 NIT Rourkela AI Agent Assistant</h1>

<p><i>A stateful, multi-threaded, tool-using RAG chatbot powered by LangGraph, Google Gemini, and Streamlit.</i></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Gemini_2.5-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
</p>

<p>
  <img src="https://img.shields.io/github/stars/Iamsaikumar14/NIT-rourkela-AI-agent-assistant?style=flat-square&color=a855f7" alt="stars"/>
  <img src="https://img.shields.io/github/forks/Iamsaikumar14/NIT-rourkela-AI-agent-assistant?style=flat-square&color=38bdf8" alt="forks"/>
  <img src="https://img.shields.io/github/last-commit/Iamsaikumar14/NIT-rourkela-AI-agent-assistant?style=flat-square&color=a855f7" alt="last commit"/>
  <img src="https://img.shields.io/github/languages/top/Iamsaikumar14/NIT-rourkela-AI-agent-assistant?style=flat-square&color=38bdf8" alt="top language"/>
</p>

</div>

---

## ✨ Overview

**NIT Rourkela AI Agent Assistant** is a conversational AI agent that combines:

- 🧠 **LangGraph** for stateful, tool-calling agent orchestration
- ✨ **Google Gemini 2.5 Flash** as the reasoning + vision LLM
- 📚 **Retrieval-Augmented Generation (RAG)** over your own uploaded documents and images
- 🔎 **Live web search** via DuckDuckGo for up-to-date answers
- 💬 A polished, dark-themed **Streamlit** chat interface with multi-conversation history

Everything — chat memory, uploaded documents, and vector embeddings — is persisted locally in **SQLite**, so the assistant remembers past conversations and indexed files across restarts, with no external database required.

<br/>

<div align="center">
<img src="https://github.com/user-attachments/assets/738f480a-a2b8-463f-b4c6-af544fd6e9ca" alt="System architecture diagram" width="100%"/>
</div>

<br/>

## 🚀 Features

| | |
|---|---|
| 💬 **Multi-threaded Chat** | Start, switch between, and delete multiple independent conversations from the sidebar |
| 🧠 **Persistent Memory** | Conversations are checkpointed to SQLite via LangGraph's `SqliteSaver`, so history survives restarts |
| 📄 **Document RAG** | Upload `.txt`, `.md`, `.pdf`, `.csv`, and `.json` files — they're chunked, embedded, and made searchable |
| 🖼️ **Image Understanding** | Upload `.png` / `.jpg` / `.jpeg` images — Gemini's vision model transcribes text, tables, and diagrams for retrieval |
| 🔎 **Live Web Search** | The agent can reach out to DuckDuckGo when it needs current information beyond its own knowledge |
| 🛠️ **Tool-calling Agent Graph** | Built as a LangGraph `StateGraph` that routes between the chat node and a `ToolNode` automatically |
| 🔍 **Source Transparency** | Every RAG answer shows the retrieved chunks, similarity scores, and source files/images used |
| 🐳 **Dockerized** | Ships with a `Dockerfile` and `docker-compose.yml` for a one-command deployment |

<br/>

## 🧩 Tech Stack

- **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph`, `ToolNode`, `SqliteSaver`)
- **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash`, `gemini-embedding-001`) via `langchain-google-genai`
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Search Tool:** `duckduckgo_search`
- **Storage:** SQLite (chat checkpoints in `chatbot.db`, document embeddings in `rag_documents.db`)
- **PDF Parsing:** `pypdf`
- **Containerization:** Docker & Docker Compose

<br/>

## 📁 Project Structure

```
NIT-rourkela-AI-agent-assistant/
├── langraphtoolfrontend.py   # Streamlit UI — chat, sidebar, document upload panel
├── langraphtoolbackend.py    # LangGraph agent — StateGraph, tools, SQLite checkpointer
├── rag_service.py            # RAG pipeline — chunking, embeddings, cosine-similarity search
├── backend.py                # Minimal standalone Gemini + LangGraph chat example (no tools)
├── test_rag.py                # Tests for the RAG service
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image definition
├── docker-compose.yml         # One-command local deployment
├── .env.example                # Sample environment file
├── chatbot.db                  # SQLite store for conversation checkpoints (auto-created)
└── rag_documents.db             # SQLite store for document chunks & embeddings (auto-created)
```

<br/>

## ⚙️ Getting Started

### Prerequisites

- Python **3.12+**
- A **Google Gemini API key** — [get one here](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/Iamsaikumar14/NIT-rourkela-AI-agent-assistant.git
cd NIT-rourkela-AI-agent-assistant
```

### 2. Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and add your API key:

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Run the app

```bash
streamlit run langraphtoolfrontend.py
```

The app will be available at **http://localhost:8501** 🎉

<br/>

## 🐳 Run with Docker

```bash
docker compose up --build
```

This builds the image from the included `Dockerfile`, mounts your `.env` file, and exposes the app on port `8501`.

<br/>

## 🧠 How It Works

1. The user's message enters a **LangGraph `StateGraph`** with a single `messages` state channel.
2. The `chatnode` calls **Gemini 2.5 Flash**, bound with two tools:
   - `search_documents` → queries the local RAG store (`rag_service.py`) for relevant chunks from uploaded files/images
   - DuckDuckGo search → fetches live information from the web
3. LangGraph's `tools_condition` automatically routes to the `ToolNode` whenever the model requests a tool call, then loops back to `chatnode` with the tool's output.
4. Every turn is checkpointed to **`chatbot.db`** via `SqliteSaver`, keyed by a per-conversation `thread_id`, so each chat thread has independent, durable memory.
5. Uploaded documents are chunked (`RecursiveCharacterTextSplitter`), embedded with `gemini-embedding-001`, and stored as float32 blobs in **`rag_documents.db`**; retrieval uses in-Python cosine similarity — no external vector database needed.

<br/>

## 🗺️ Roadmap Ideas

- [ ] Add a real screenshot / demo GIF to this README
- [ ] Swap the manual cosine-similarity scan for an indexed vector search (e.g. `sqlite-vec`, FAISS) as documents scale
- [ ] Add authentication for multi-user deployments
- [ ] Add automated CI (lint + `test_rag.py`) via GitHub Actions

<br/>

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Iamsaikumar14/NIT-rourkela-AI-agent-assistant/issues).

## 📄 License

No license file is currently included in this repository. Consider adding one (e.g. MIT) if you plan to accept external contributions.

---

<div align="center">
<sub>Built with ❤️ using LangGraph, Gemini, and Streamlit</sub>
</div>
