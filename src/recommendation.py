import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load the cleaned data
def load_data(file_path):
    return pd.read_csv(file_path)

# Vectorize the cleaned text (we'll use TF-IDF)
def vectorize_text(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])  # Correct column name here
    return tfidf_matrix, vectorizer



# Get recommendations based on course ID (or title)
def get_recommendations(course_id, df, tfidf_matrix):
    # Check if the course_id exists in the dataset
    if course_id not in df['course_id'].values:
        print(f"Course ID {course_id} not found in the dataset.")
        print("Try following course ids instead")
        print(df[['course_id', 'subject']].sample(5))
        return None
    
    # Find the index of the given course
    idx = df[df['course_id'] == course_id].index[0]
    
    # Compute cosine similarity between the given course and all other courses
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get the top 5 most similar courses (excluding the course itself)
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]
    
    recommended_courses = df.iloc[similar_indices]
    return recommended_courses[['course_id', 'course_title', 'subject']]

# Main execution
if __name__ == "__main__":
    # Load cleaned data
    df = load_data('artifacts/cleaned_udemy_courses.csv')
    
    # Vectorize the cleaned text
    tfidf_matrix, vectorizer = vectorize_text(df)
    
    # Example: Get recommendations for a course with course_id = 1
    course_id = 1 # Replace with any course_id you'd like to test
    recommended_courses = get_recommendations(course_id, df, tfidf_matrix)
    
    if recommended_courses is not None:
        print(f"Recommended courses for course ID {course_id}:")
        print(recommended_courses)



