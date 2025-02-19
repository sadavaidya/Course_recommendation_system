from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib  # To save the vectorizer



def extract_features(df):
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply the vectorizer on the combined text (course_title + subject)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
    
    # Return the TF-IDF matrix and the vectorizer for later use
    return tfidf_matrix, tfidf_vectorizer



def save_vectorizer(tfidf_vectorizer, filename="tfidf_vectorizer.pkl"):
    joblib.dump(tfidf_vectorizer, filename)



def main():
    # Load the cleaned data (assuming the data is cleaned and stored in 'data_preprocessing.py')
    df = pd.read_csv('artifacts/cleaned_udemy_courses.csv')  # Update the path as necessary

    # Extract features from the combined text
    tfidf_matrix, tfidf_vectorizer = extract_features(df)
    
    # Save the vectorizer for later use
    save_vectorizer(tfidf_vectorizer)

    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Vectorizer saved as tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()
