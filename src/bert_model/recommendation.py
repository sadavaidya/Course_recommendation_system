import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load course dataset
df = pd.read_csv("artifacts/cleaned_udemy_courses.csv")  # Ensure it has a column 'course_title'

# Load stored embeddings and FAISS index
course_embeddings = np.load("src/bert_model/bert_embeddings.npy")
index = faiss.read_index("src/bert_model/faiss_index.bin")

def recommend_courses(user_query, top_k=5):
    """Given a user query, return the top K recommended courses."""
    query_embedding = model.encode([user_query], normalize_embeddings=True)
    
    # Search in FAISS index
    _, indices = index.search(query_embedding, top_k)
    
    # Retrieve recommended courses
    recommended_courses = df.iloc[indices[0]]["course_title"].tolist()
    
    return recommended_courses

# Take user input
user_query = input("Enter your course interest: ")
recommendations = recommend_courses(user_query)

# Display recommendations
print("\nTop 5 Recommended Courses:")
for i, course in enumerate(recommendations, 1):
    print(f"{i}. {course}")
