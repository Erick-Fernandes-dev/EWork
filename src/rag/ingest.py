# src/rag/ingest.py
import os
import pickle
import faiss
from pypdf import PdfReader
# CORREÇÃO: A importação agora vem do novo pacote
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PDF_PATH = "docs/specs.pdf"
DB_PATH = "db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_data():
    """Lê o PDF, cria chunks, embeddings e salva um índice FAISS."""
    print("Starting data ingestion...")

    # 1. Ler o PDF
    reader = PdfReader(PDF_PATH)
    documents = []
    for i, page in enumerate(reader.pages):
        documents.append({
            "text": page.extract_text(),
            "page": i + 1
        })
    print(f"Loaded {len(documents)} pages from PDF.")

    # 2. Fazer o chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for text in split_texts:
            chunks.append({
                "snippet": text,
                "page": doc["page"]
            })
    print(f"Created {len(chunks)} text chunks.")

    # 3. Gerar embeddings
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([chunk["snippet"] for chunk in chunks], show_progress_bar=True)

    # 4. Criar e salvar o índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(DB_PATH, exist_ok=True)
    faiss.write_index(index, f"{DB_PATH}/vector_index.faiss")
    print(f"FAISS index saved to {DB_PATH}/vector_index.faiss")

    # 5. Salvar os chunks para recuperação posterior
    with open(f"{DB_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {DB_PATH}/chunks.pkl")

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()