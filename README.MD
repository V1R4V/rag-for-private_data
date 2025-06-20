# 📚 RAG Model Using Private Data

**RAG Model Using Private Data** is a modular Retrieval-Augmented Generation (RAG) pipeline built for enterprise-scale document understanding. It uses [Docling](https://github.com/IBM/docling) to extract structured information from complex, multi-format documents — a critical first step in preparing private data for large language model (LLM) workflows.

---

## Prerequisites

1. Install the required packages:
``pip install -r requirements.txt``
2. Set up your environment variables by creating a .env file:

## 🚀 Key Features

- 🔄 **Multi-format Parsing**: Seamlessly handles PDFs, DOCX, PPTX, Markdown, HTML, and more.
- 🧠 **AI-Powered Document Understanding**:
  - Layout analysis with RT-DETR
  - Table recognition via TableFormer
- 📊 **Chunking Engine**: Smart, hierarchical chunking optimized for embedding in RAG pipelines.
- 📦 **Ideal for Enterprise GenAI**: Enables knowledge ingestion, retrieval, and reasoning over internal documents.

---

## 📁 Project Structure (Current Phase)

```bash
rag-private-data/
├── parser/
│   └── docling_parser.py     # Parser using Docling
├── data/
│   └── sample.docx           # Example private document
├── .env                      # API keys and secrets
├── requirements.txt
└── README.md
