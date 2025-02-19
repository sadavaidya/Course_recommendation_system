import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned dataset
df = pd.read_csv('artifacts/cleaned_udemy_courses.csv')

# Load the saved TF-IDF vectorizer and matrix
with open('artifacts/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('artifacts/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

# Function to recommend courses
def recommend_courses(query, top_n=10):
    query_vector = tfidf_vectorizer.transform([query])  # Convert input text to vector
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)  # Compute similarity
    top_indices = cosine_sim.argsort()[0][-top_n:][::-1]  # Get top matches

    return df.iloc[top_indices][['course_title', 'subject', 'url']]

# Streamlit UI
st.title("Course Recommendation System")
st.write("Enter a keyword related to your interest, and we'll suggest relevant courses.")

# User input
user_input = st.text_input("Enter a keyword:")

if user_input:
    recommendations = recommend_courses(user_input)
    st.write("### Recommended Courses:")
    for _, row in recommendations.iterrows():
        st.write(f"**{row['course_title']}** ({row['subject']})")
        st.markdown(f"[Course Link]({row['url']})")

