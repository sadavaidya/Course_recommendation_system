import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned data
df = pd.read_csv('artifacts/cleaned_udemy_courses.csv')

# Ensure column name is correct
if 'combined_text' not in df.columns:
    print("Error: 'combined_text' column not found in the dataset.")
    exit()

# Initialize and fit TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

# Save TF-IDF Vectorizer
with open('artifacts/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Save TF-IDF Matrix
with open('artifacts/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

print("TF-IDF Vectorizer and Matrix saved successfully!")
