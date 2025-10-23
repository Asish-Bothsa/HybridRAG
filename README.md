# 🧠 HybridRAG
**HybridRAG: A Multi-Source Retrieval-Augmented Generation Framework using Vector and Graph Databases**

---

## 📘 Overview
**HybridRAG** is an intelligent chat and reasoning system that combines the power of **semantic vector search (Pinecone)** and **graph-based knowledge reasoning (Neo4j)**.  
It enables **context-rich, grounded, and explainable** conversations by retrieving both unstructured and structured knowledge — bridging the gap between traditional RAG and knowledge graphs.

---

## 🚀 Key Features
- 🔍 **Dual Retrieval Engine** — Combines Pinecone vector search with Neo4j graph queries for multi-source context.
- 🧩 **Context Fusion** — Integrates results into a unified reasoning prompt for the LLM.
- 🧠 **Explainable Answers** — Outputs concise responses with reasoning and graph node references.
- ⚡ **Asynchronous Architecture** — Fetches vector and graph context concurrently for speed.
- 🧾 **Scalable Backend** — Modular design ready for deployment or fine-tuned expansion.
- 🗣️ **Interactive Chat Mode** — Terminal-based conversational interface with progress feedback.

---

## 🏗️ Architecture

         ┌──────────────────────────────┐
         │          User Query          │
         └──────────────┬───────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │  Dual Retrieval Engine  │
          │ ┌──────────────┐        │
          │ │  Pinecone    │─semantic context
          │ └──────────────┘        │
          │ ┌──────────────┐        │
          │ │   Neo4j      │─graph facts
          │ └──────────────┘        │
          └──────────────┬──────────┘
                         │
                         ▼
          ┌─────────────────────────┐
          │   Context Fusion Layer   │
          │   (build_prompt logic)   │
          └──────────────┬───────────┘
                         │
                         ▼
          ┌─────────────────────────┐
          │     LLM Inference       │
          │ (OpenAI / local model)  │
          └──────────────┬───────────┘
                         │
                         ▼
          ┌─────────────────────────┐
          │     Response Output      │
          │ (with reasoning ) │
          └─────────────────────────┘

          
---
## Graph Database
<p align="center">
  <img src="https://github.com/user-attachments/assets/238c1e7c-4f9d-4e5a-901c-9803a0bf92dc" alt="Graph database" width="700"/>
</p>


## ⚙️ Tech Stack
| Component | Technology |
|------------|-------------|
| LLM Backend | OpenAI GPT models (configurable) |
| Vector DB | Pinecone |
| Graph DB | Neo4j |
| Language | Python 3.10+ |
| Env Handling | python-dotenv |
| Async Framework | asyncio |
| CLI Interface | Rich |

---

## 🧰 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/HybridRAG.git
cd HybridRAG
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv .venv
```

###3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Setup Environment Variables
```bash
open .env and fill in your API credentials:
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
````
### 5️⃣ Run the Application
```bash
python src/hybrid_chat.py
```
## Sample Output
<p align="center">
  <img src="https://github.com/user-attachments/assets/49ae19bd-0f16-4382-9764-b7bbbd5f538d " width="700" />
</p>

📜 License

Distributed under the MIT License. See LICENSE
 for more information.

 🌟 Acknowledgements

Pinecone Vector DB

Neo4j Graph Database

OpenAI API




