# src/build_index_safe.py
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import gc
from scipy import sparse

# Paths
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = PROCESSED_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
MAX_FEATURES = 20000 
SVD_DIM = 400 #trying to optimize relevance 
N_LIST = 100
SAMPLE_SIZE = 500000  # number of samples to fit TF-IDF and SVD
BATCH_SIZE = 50000  # number of rows to process at a time

# Main
if __name__ == "__main__":
    print("Listing all Parquet chunks...")
    chunk_files = sorted(PROCESSED_DIR.glob("arxiv_chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError("No parquet chunks found. Run preprocess_faiss.py first!")

    # Step 1: Fit TF-IDF and SVD on a representative sample
    print("Fitting TF-IDF and SVD on a representative sample...")
    sample_chunks = []
    for f in chunk_files:
        df = pd.read_parquet(f, columns=["title", "clean_abstract"])
        sample_chunks.append(df)
        if sum(len(c) for c in sample_chunks) >= SAMPLE_SIZE:
            break
    sample_df = pd.concat(sample_chunks).sample(SAMPLE_SIZE, random_state=42)
    sample_texts = (sample_df["title"].fillna("") + " " + sample_df["clean_abstract"].fillna(""))

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_sample = vectorizer.fit_transform(sample_texts)

    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    X_reduced_sample = svd.fit_transform(X_sample).astype("float32")

    # Step 2: Train FAISS index
    print("Training FAISS index...")
    faiss.normalize_L2(X_reduced_sample)
    quantizer = faiss.IndexFlatIP(SVD_DIM)
    index = faiss.IndexIVFFlat(quantizer, SVD_DIM, N_LIST, faiss.METRIC_INNER_PRODUCT)
    index.train(X_reduced_sample)

    # Step 3: Add vectors in batches
    total_vectors = 0
    for i, f in enumerate(chunk_files, 1):
        print(f"Processing chunk {i}/{len(chunk_files)}: {f.name}")
        df_chunk = pd.read_parquet(f, columns=["title", "clean_abstract"])
        texts = (df_chunk["title"].fillna("") + " " + df_chunk["clean_abstract"].fillna(""))

        # Process in smaller batches to save memory
        for start in range(0, len(df_chunk), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(df_chunk))
            batch_texts = texts.iloc[start:end]
            X_batch = vectorizer.transform(batch_texts)
            X_reduced_batch = svd.transform(X_batch).astype("float32")
            faiss.normalize_L2(X_reduced_batch)
            index.add(X_reduced_batch)
            total_vectors += len(batch_texts)
            print(f"  → Added batch {start}-{end-1}, total vectors = {total_vectors}")
            del X_batch, X_reduced_batch
            gc.collect()  # free memory after each batch

    # Step 4: Save all artifacts
    print("\nCleaning up memory before saving artifacts...")
    gc.collect()

    print("Saving FAISS index (this may take several minutes)...")
    faiss.write_index(index, str(OUT_DIR / "faiss_index_ivf.idx"))

    print("Saving TF-IDF vectorizer and SVD transformer...")
    joblib.dump(vectorizer, OUT_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(svd, OUT_DIR / "svd_transformer.joblib")

    # Combine metadata
    print("Combining all metadata into a single parquet file...")
    meta_cols = ["id", "title", "categories", "authors", "abstract"]
    all_meta = pd.concat([pd.read_parquet(f, columns=meta_cols) for f in chunk_files])
    all_meta.to_parquet(OUT_DIR / "meta.parquet", index=False)

    print("\nAll artifacts saved successfully to data/processed/")
