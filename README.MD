# 📚 RAG Assistant for Private Enterprise Data

**RAG Assistant** is a modular, production-grade Retrieval-Augmented Generation (RAG) pipeline tailored for private enterprise documents like RFPs, RFIs, whitepapers, and contracts. The system uses intelligent document conversion, semantic embedding, and a local vector store (ChromaDB) to enable intelligent document Q\&A powered by LLMs (via Ollama or HuggingFace).
### Note ⚠: While this code mirrors the functionality and structure of the original project, the actual implementation has been adapted to exclude confidential assets. It is intended to demonstrate technical understanding and design ability without breaching data privacy agreements.

---

## 🌍 Why This Project?

In the world of automation, I felt some tasks can be easily automated like filling RFI/RFP's and Drafting pitch-decks

* Real-world parsing of enterprise documents
* Vector storage with semantic search
* Interfacing multiple LLM types (local + cloud)
* Answering domain-specific questions with retrieved context

---

## ⚡ Key Features

* ✅ **Multi-format support**: Handles Excel, PDF, Word (via Docling)
* 🧠 **Semantic Retrieval**: Uses Sentence-Transformer for vector embeddings
* 🏑 **Flexible LLM Integration**:

  * 🤖 Ollama (offline inference via LLaMA)
  * 💡 HuggingFace (optional, free alternative)
* 🔐 **Private-by-default**: No external calls unless you choose to
* ⏳ **Interactive CLI**: Real-time document question answering

---

## 📂 Directory Structure

```bash
rag-assistant/
├── rag_rfp_assistant.py         # Main class + pipeline logic
├── data/
│   └── sample_rfi.xlsx          # Sample enterprise RFI
├── requirements.txt
├── README.md
```

---

## 📆 Setup Instructions

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Pull LLM model (if using Ollama)

```bash
ollama pull llama3:latest
ollama serve  # Ensure Ollama server is running
```

### 3. Run the Assistant

```bash
python rag_rfp_assistant.py
```

You can also use it modularly in your code:

```python
from rag_rfp_assistant import RAGRFPFiller
rag = RAGRFPFiller(llm_type="ollama")
rag.initialize_components()
chunks = rag.process_document("data/sample_rfi.xlsx")
rag.store_chunks(chunks)
rag.answer_rfp_question("How many offices does the company have?")
```

---

## 🔄 Workflow Summary

```text
+--------------------------+
|  Raw RFI / RFP Document |
+--------------------------+
            |
            v
+--------------------------+
|  Structured Markdown     |
|  via Docling Parser      |
+--------------------------+
            |
            v
+--------------------------+
|  Q&A Chunk Extraction    |
+--------------------------+
            |
            v
+--------------------------+
|  Vector Embedding        |
|  (MiniLM or INSTRUCTOR)  |
+--------------------------+
            |
            v
+--------------------------+
|  Store in ChromaDB       |
+--------------------------+
            |
            v
+--------------------------+
|  Query + Retrieval       |
+--------------------------+
            |
            v
+--------------------------+
|  Context-aware Response  |
|  via LLM (Ollama/ HF)    |
+--------------------------+
```

---

## 🔎 Example Interaction

```
> What is the legal name of the company?
VDart Inc

> How many US locations does it operate?
1.0 (based on retrieved RFI chunk)
```

---

## 💼 Technologies Used

* Python 3.10+
* [Docling](https://github.com/IBM/docling)
* [ChromaDB](https://www.trychroma.com/)
* [Sentence-Transformers](https://www.sbert.net/)
* [Ollama](https://ollama.ai/) / HuggingFace Transformers

---

## 🚀 Future Enhancements

* [ ] Web UI with Streamlit or Flask
* [ ] Retrieval scoring and ranking
* [ ] Add authentication for internal usage
* [ ] Support real-time document ingestion via API

---

## 🎓 Learning Outcomes

This project showcases your ability to:

* Understand document parsing and NLP pipelines
* Integrate vector stores and embeddings
* Use modern LLMs responsibly and efficiently
* Write reusable, modular code for production-like tasks

---

## 📗 Attribution

This public version is a generalization of an enterprise use case developed under NDA. No proprietary content or identifiers are included.

*Built with learning, ethics, and AI.*
