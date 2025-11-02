
# AI Medical Chatbot with RAG 🩺

An open-source AI Medical Chatbot built with Retrieval-Augmented Generation (RAG), combining HuggingFace embeddings, FAISS, and Groq LLMs to deliver accurate, reference-backed medical answers through an intuitive Streamlit interface.

## 🧠 Overview

**AI Medical Chatbot for Healthcare** is an end-to-end open-source project that demonstrates how to build a **production-grade medical assistant** using **Retrieval-Augmented Generation (RAG)**.

The system allows users to ask complex medical questions and receive **accurate, evidence-backed answers** grounded in a **curated knowledge base** (such as medical textbooks or research PDFs).

Instead of hallucinating, the chatbot retrieves relevant document chunks from a **FAISS vector database** built using **HuggingFace sentence embeddings**, then passes them to a **large language model** (hosted on **Groq** or **HuggingFace**) for context-aware reasoning and summarization.

The project includes both a **command-line prototype** and an intuitive **Streamlit web interface**, making it easy for researchers, healthcare developers, and AI enthusiasts to explore trustworthy AI in healthcare.


---

## ✨ Key Features

- **🔍 Retrieval-Augmented Generation (RAG) Pipeline** – Combines dense retrieval via FAISS with LLM reasoning for context-aware answers.  
- **📘 Curated Medical Knowledge Base** – Answers are grounded in real medical literature or uploaded PDFs, ensuring factual reliability.  
- **⚡ FAISS Vector Store** – Enables high-speed semantic search across thousands of medical text chunks.  
- **🧠 HuggingFace Sentence Embeddings** – Uses `all-MiniLM-L6-v2` for efficient and high-quality vector representations (switchable to remote API).  
- **🤖 Multiple LLM Backends** – Compatible with **Groq (Llama 4 Maverick)** and **HuggingFace Inference** models (e.g., Mistral 7B Instruct).  
- **🧩 Modular Architecture** – Plug-and-play design with easily swappable embedding models, vector stores, and prompt templates.  
- **💬 Dual Interfaces** –  
  - `connect_memory_with_llm.py`: CLI prototype for quick testing.  
  - `medibot.py`: Streamlit UI for conversational interaction.  
- **🧾 Source Traceability** – Every answer includes document references showing which chunks supported the response.  
- **⚙️ Caching & Optimization** – Streamlit resource caching for faster embedding and vector retrieval.  
- **🪶 100% Open-Source Stack** – Built entirely with open tools and frameworks for reproducibility and extensibility.  

---

## 🏗️ Technical Architecture

The **AI Medical Chatbot for Healthcare** is built on a modular **Retrieval-Augmented Generation (RAG)** pipeline designed for **accuracy**, **scalability**, and **transparency**.  
It ensures all responses are grounded in real medical knowledge by combining semantic search with large language model reasoning.

### 🔹 Architecture Flow

![Technical Architecture](./TECHNICAL%20IMAGE.png)

```
📄 Medical PDFs
│
▼
🔹 Text Extraction & Chunking
│
▼
🔹 Embedding Generation (HuggingFace: all-MiniLM-L6-v2)
│
▼
🔹 FAISS Vector Index (vectorstore/db_faiss)
│
▼
💬 User Query ───────────────┐
▼
🔹 Retriever (Top-K Semantic Search)
│
▼
🔹 Prompt Assembly (RAG Context)
│
▼
🤖 LLM Generation (Groq or HuggingFace)
│
▼
🧠 Final Answer + Source References
```
---
### 🔹 Workflow Breakdown

1. **Data Extraction & Preparation**  
   - Raw medical PDFs are processed, cleaned, and segmented into logical text chunks.  
   - This granularity allows precise retrieval of relevant medical content.

2. **Embedding & Storage**  
   - Each text chunk is encoded into dense vector embeddings using **HuggingFace SentenceTransformer (`all-MiniLM-L6-v2`)**.  
   - Vectors are indexed in a **FAISS** database for high-speed semantic search.

3. **Query Processing**  
   - The user submits a medical question via the **Streamlit UI**.  
   - A retriever fetches the **top-K most relevant chunks** from the FAISS index.  
   - These retrieved contexts are dynamically assembled into a **prompt** and passed to the **LLM**.

4. **Response Generation**  
   - The **LLM (Groq-hosted Llama 4 Maverick or HuggingFace model)** generates an answer based strictly on retrieved context.  
   - The chatbot outputs both the **final answer** and **source document references**, ensuring factual transparency.

5. **Modular & Extensible Design**  
   - Embedding models, LLM providers, or PDF knowledge bases can be easily swapped or upgraded.  
   - Streamlit caching optimizes repeated queries for faster response times.

---

### 🔹 Key Advantages

- Reliable, **reference-backed responses** (no hallucinations).  
- **Plug-and-play architecture** for rapid experimentation.  
- Designed for **healthcare-grade transparency and maintainability**.  

---

### Main Components
| File | Role |
|------|------|
| `create_memory_for_llm.py` | Builds FAISS index from PDFs (embedding + persist) |
| `connect_memory_with_llm.py` | CLI RAG query using HuggingFaceEndpoint |
| `medibot.py` | Streamlit chat interface using Groq Chat model + FAISS retrieval |
| `vectorstore/db_faiss` | Persisted FAISS index (created beforehand) |
| `data/` | PDF source documents |

---

## 🧭 Clone the Project
To get started, clone this repository to your local machine using **Git**:

```bash
git clone https://github.com/AdritPal08/AI-Medical-Chatbot-with-RAG.git
```
Then navigate to the project directory:
```bash
cd AI-Medical-Chatbot-with-RAG
```
## 🐍 Setting Up a Python Virtual Environment

You can use **Pipenv**, **pip + venv**, **Conda**, or **UV** depending on your preference.

---

### ⚙️ Using Pipenv

1. **Install Pipenv (if not already installed):**
```bash
   pip install pipenv
```
2. **Install Dependencies with Pipenv:**
```bash
   pipenv install
```
3. **Activate the Virtual Environment:**
```bash
   pipenv shell
```

### 🧱 Using pip and venv

1. **Create a Virtual Environment:**
```bash
   python -m venv venv
```
2. **Activate the Virtual Environment:**
- ***macOS/Linux:***
```bash
   source venv/bin/activate
```
- ***Windows:***
```bash
   venv\Scripts\activate
```
3. **Install Dependencies:**
```bash
   pip install -r requirements.txt
```

### 🧬 Using Conda

1. **Create a Conda Environment:**
```bash
   conda create --name myenv python=3.11
```
2. **Activate the Conda Environment:**
```bash
   conda activate myenv
```
3. **Install Dependencies:**
```bash
   pip install -r requirements.txt
```

### ⚡ Using UV (Fast Python Package Manager)

1. **Install UV:**
```bash
   pip install uv
```
2. **Create a Virtual Environment:**
```bash
   uv venv
```
3. **Activate the Virtual Environment:**
- ***macOS/Linux:***
```bash
   source .venv/bin/activate
```
- ***Windows:***
```bash
   .venv\Scripts\activate
```
4. **Install Dependencies:**
```bash
   uv pip install -r requirements.txt
```

## 🤝 **Contributing**
Contributions are welcome! If you have ideas, improvements, or bug fixes, feel free to **open an issue** or **submit a pull request**.

## 📜 **License**
This project is licensed under the **[GNU GENERAL PUBLIC LICENSE Version 3](LICENSE)**.

## ⭐ **Support & Feedback**
If you like this project, don't forget to ⭐ star the repository!

For feedback and inquiries, contact **[LinkedIn](https://www.linkedin.com/in/adritpal/)**.


---
Developed by **Adrit Pal** 🚀

