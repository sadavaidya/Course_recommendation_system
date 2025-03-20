import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load TF-IDF vectorizer and matrix
with open("src/tfidf_model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("src/tfidf_model/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# Load SBERT embeddings
with open("src/bert_model/bert_embeddings.pkl", "rb") as f:
    sbert_embeddings = pickle.load(f)

# Compute cosine similarity matrices
tfidf_sim = cosine_similarity(tfidf_matrix)
sbert_sim = cosine_similarity(sbert_embeddings)

# Extract upper triangle (excluding diagonal) to remove self-similarity
tfidf_sim_values = tfidf_sim[np.triu_indices_from(tfidf_sim, k=1)]
sbert_sim_values = sbert_sim[np.triu_indices_from(sbert_sim, k=1)]

# Plot histograms
plt.figure(figsize=(10, 5))
sns.histplot(tfidf_sim_values, bins=50, color="blue", label="TF-IDF", kde=True)
sns.histplot(sbert_sim_values, bins=50, color="orange", label="SBERT", kde=True)
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Frequency")
plt.title("Cosine Similarity Distribution: TF-IDF vs. SBERT")
plt.legend()

# Save the figure as a PNG file
plt.savefig("artifacts/cosine_similarity_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

