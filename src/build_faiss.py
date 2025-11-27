# src/build_faiss.py

import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

TICKETS_PATH = "data/tickets.csv"
INDEX_DIR = "models/faiss"
INDEX_FILE = os.path.join(INDEX_DIR, "tickets.index")
EMB_FILE = os.path.join(INDEX_DIR, "embeddings.npy")
MAP_FILE = os.path.join(INDEX_DIR, "mapping.csv")

def build_faiss_index():
    if not os.path.exists(TICKETS_PATH):
        raise FileNotFoundError(f"{TICKETS_PATH} not found — run preprocess_multimodal.py first.")

    print("[FAISS] Loading tickets...")
    tickets = pd.read_csv(TICKETS_PATH)

    tickets['combined'] = (
        tickets['title'].astype(str) + " | " +
        tickets['description'].astype(str) + " | " +
        tickets['remediation'].astype(str)
    )

    print("[FAISS] Loading embedding model (SentenceTransformer) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[FAISS] Generating embeddings...")
    embeddings = model.encode(
        tickets['combined'].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )

    dim = embeddings.shape[1]
    
    print(f"[FAISS] Creating index of dimension {dim} ...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)

    print("[FAISS] Saving index and metadata...")
    faiss.write_index(index, INDEX_FILE)
    np.save(EMB_FILE, embeddings)
    tickets.to_csv(MAP_FILE, index=False)

    print("\n[✔] FAISS index built successfully!")
    print(f"[✔] Index saved to: {INDEX_FILE}")
    print(f"[✔] Embeddings saved to: {EMB_FILE}")
    print(f"[✔] Mapping saved to: {MAP_FILE}")

if __name__ == "__main__":
    build_faiss_index()
