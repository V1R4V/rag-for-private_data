from docling.document_converter import DocumentConverter
from InstructorEmbedding import INSTRUCTOR
import chromadb
import os
import traceback

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
for i, chunk in enumerate(chunks[:10]):
    print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

# Save to file for vector storage later
with open("chunks_rfi.txt", "w") as f:
    for c in chunks:
        f.write(c + "\n\n---\n\n")
print(f"\n Total chunks extracted: {len(chunks)}")





