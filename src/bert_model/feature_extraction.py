import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & fast

# Load your dataset (assuming CSV with 'course_description' column)
df = pd.read_csv("artifacts/cleaned_udemy_courses.csv")

# Generate SBERT embeddings
course_embeddings = model.encode(df["course_title"].tolist(), normalize_embeddings=True)

# Save embeddings for future use
np.save("src/bert_model/bert_embeddings.npy", course_embeddings)
with open('src/bert_model/bert_embeddings.pkl', 'wb') as f:
    pickle.dump(course_embeddings, f)

# Dimensions of embeddings
embedding_dim = course_embeddings.shape[1]

# Create FAISS index (L2-normalized for cosine similarity)
index = faiss.IndexFlatIP(embedding_dim)  
index.add(course_embeddings)  # Add embeddings to the index

# Save FAISS index
faiss.write_index(index, "src/bert_model/faiss_index.bin")
