from docling.document_converter import DocumentConverter
from InstructorEmbedding import INSTRUCTOR
import chromadb
import os
import traceback
from sentence_transformers import SentenceTransformer
from pprint import pprint
import time

start_time = time.time()
# Load and convert
source = "/Users/vibhravjha/Code/VSCODE/AI_AGENT_BASIC/PDFS/RFI 2025.xlsx"
converter = DocumentConverter()
result = converter.convert(source)

# Export to markdown (table-style)
markdown_text = result.document.export_to_markdown()

# Chunk each Question-Answer pair from markdown table rows
def chunk_markdown_table_rows(md_text):
    chunks = []
    lines = md_text.splitlines()

    # Skip header and separator lines
    rows = [line for line in lines if '|' in line and not line.startswith('| ---')]

    for row in rows[1:]:  # Skip the header row
        columns = [col.strip() for col in row.strip().split('|')[1:-1]]  # Remove empty split ends
        if len(columns) >= 2:
            question, answer = columns[0], columns[1]
            if question and answer:
                chunk = f"### Question\n{question}\n\n### Answer\n{answer}"
                chunks.append(chunk)
    return chunks

# Create row chunks
chunks = chunk_markdown_table_rows(markdown_text)

# Preview first 5 chunks
'''for i, chunk in enumerate(chunks[:10]):
    print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

# Save to file for vector storage later
with open("chunks_rfi.txt", "w") as f:
    for c in chunks:
        f.write(c + "\n\n---\n\n")
print(f"\n Total chunks extracted: {len(chunks)}")'''



try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded successfully")
except Exception as e:
    print("Error loading the embedding model:")
    traceback.print_exc()
    exit(1)

try:
    client = chromadb.EphemeralClient() 
    collection = client.get_or_create_collection("rfi_chunks")
    print("ChromaDB initialized successfully")
except Exception as e:
    print("Error initializing ChromaDB or creating collection:")
    traceback.print_exc()
    exit(1)

print("\nInserting chunks into ChromaDB")

# logic for inserting chunks into chromaDB
for i, chunk in enumerate(chunks):
    try:
        embedding = model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk-{i}"],
            metadatas=[{"source": "RFI 2025.xlsx"}]
        )
        
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(chunks)} chunks")
            
    except Exception as e:
        print(f"Error processing chunk {i}:")
        traceback.print_exc()
        continue 
print(f"All {len(chunks)} chunks inserted successfully")

#Testing for query retrival 
results = collection.query( 
    query_texts=[ "name of company?"],
    n_results=2

)
pprint(results)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")





