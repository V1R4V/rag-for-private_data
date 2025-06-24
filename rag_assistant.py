import os
import time
import json
import torch
import requests
import traceback
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter
import chromadb
from InstructorEmbedding import INSTRUCTOR


class RAGPrivateDocAssistant:
    def __init__(self, 
                 llm_type="ollama", 
                 ollama_host="http://localhost:11434", 
                 model_name="llama3.2", 
                 hf_model_name="microsoft/DialoGPT-medium"):

        self.llm_type = llm_type
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.hf_model_name = hf_model_name

        self.embedding_model = None
        self.client = None
        self.collection = None
        self.hf_tokenizer = None
        self.hf_model = None

        self.system_prompt = (
            "You are a helpful assistant trained on enterprise documents.\n"
            "Answer clearly and directly.\n"
            "If context is insufficient, infer based on available data.\n"
            "Maintain professional tone and avoid hallucination.\n"
        )

    def initialize_components(self):
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded successfully.")
        except Exception:
            print("Failed to load embedding model.")
            traceback.print_exc()
            return False

        try:
            self.client = chromadb.EphemeralClient()
            self.collection = self.client.get_or_create_collection("doc_chunks")
            print("ChromaDB client and collection initialized.")
        except Exception:
            print("Failed to initialize ChromaDB.")
            traceback.print_exc()
            return False

        if self.llm_type == "huggingface":
            return self.initialize_huggingface()
        elif self.llm_type == "ollama":
            return self.check_ollama_connection()

        return True

    def initialize_huggingface(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self.hf_model = AutoModelForCausalLM.from_pretrained(self.hf_model_name)

            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

            print("Hugging Face model loaded.")
            return True
        except Exception as e:
            print(f"Hugging Face initialization error: {e}")
            return False

    def check_ollama_connection(self):
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if any(self.model_name in name for name in model_names):
                    print(f"Ollama model '{self.model_name}' is available.")
                    return True
                else:
                    print(f"Model '{self.model_name}' not found. Use 'ollama pull {self.model_name}'.")
                    return False
            else:
                print(f"Ollama error: Status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("Could not connect to Ollama server. Ensure it's running.")
            return False

    def convert_document(self, filepath):
        try:
            print(f"Converting file: {filepath}")
            converter = DocumentConverter()
            result = converter.convert(filepath)
            markdown_text = result.document.export_to_markdown()
            print("Conversion successful.")
            return markdown_text
        except Exception as e:
            print(f"Error converting document: {e}")
            return ""

    def extract_chunks(self, markdown_text):
        chunks = []
        lines = markdown_text.splitlines()
        rows = [line for line in lines if '|' in line and not line.startswith('| ---')]

        for row in rows[1:]:
            columns = [col.strip() for col in row.strip().split('|')[1:-1]]
            if len(columns) >= 2:
                question, answer = columns[0], columns[1]
                if question and answer:
                    chunk = f"### Question\n{question}\n\n### Answer\n{answer}"
                    chunks.append(chunk)

        print(f"Extracted {len(chunks)} chunks.")
        return chunks

    def store_chunks(self, chunks):
        if not self.embedding_model or not self.collection:
            raise ValueError("Model and ChromaDB must be initialized.")

        for i, chunk in enumerate(chunks):
            try:
                embedding = self.embedding_model.encode(chunk).tolist()
                self.collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    ids=[f"chunk-{i}"],
                    metadatas=[{"source": "doc", "chunk_id": i}]
                )
            except Exception as e:
                print(f"Error storing chunk {i}: {e}")
                continue

        print(f"Stored {len(chunks)} chunks in ChromaDB.")

    def retrieve_context(self, query, top_k=5):
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            documents = results.get("documents", [[]])[0]
            context = "\n\n---\n\n".join(documents)
            return context, results
        except Exception as e:
            print(f"Context retrieval error: {e}")
            return "", None

    def generate_response(self, query, context):
        if self.llm_type == "ollama":
            return self.query_ollama(query, context)
        else:
            return "LLM not configured."

    def query_ollama(self, query, context):
        prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "top_p": 0.9,
                        "num_predict": 500
                    }
                },
                timeout=260
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Ollama error {response.status_code}"
        except requests.exceptions.Timeout:
            return "Ollama request timed out."
        except Exception as e:
            return f"Ollama error: {e}"

    def answer_question(self, question):
        print(f"Processing question: {question}")
        context, results = self.retrieve_context(question)
        if not context:
            return "No relevant context found."

        response = self.generate_response(question, context)
        return {
            "question": question,
            "answer": response,
            "context_used": context,
            "chunks_found": len(results['documents'][0]) if results else 0
        }

    def interactive_mode(self):
        print("RAG Document Assistant Interactive Mode")
        print("Type 'quit' to exit")
        while True:
            try:
                question = input("Question: ").strip()
                if question.lower() in {"quit", "exit"}:
                    break
                if not question:
                    continue
                result = self.answer_question(question)
                print(f"Answer: {result['answer']}\n")
            except KeyboardInterrupt:
                break


def main():
    document_path = "./sample_document.docx"
    rag = RAGPrivateDocAssistant()
    if not rag.initialize_components():
        return

    markdown = rag.convert_document(document_path)
    if not markdown:
        return

    chunks = rag.extract_chunks(markdown)
    rag.store_chunks(chunks)

    result = rag.answer_question("What services does the company offer?")
    print(f"\nQuery Result:\n{result['answer']}\n")
    rag.interactive_mode()


if __name__ == "__main__":
    main()
