import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Ensure stopwords and tokenizer are available
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and tokenizes text data."""
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

def load_and_clean_data(filepath):
    """Loads dataset and cleans course data."""
    df = pd.read_csv(filepath)
    
    # Combine 'course_title' and 'subject' into a new column
    df['combined_text'] = df['course_title'] + " " + df['subject']
    
    # Apply text preprocessing
    df['clean_content'] = df['combined_text'].apply(preprocess_text)
    df.to_csv('artifacts/cleaned_udemy_courses.csv', index=False)
    return df[['course_id', 'course_title', 'subject', 'clean_content']]

if __name__ == "__main__":
    df = load_and_clean_data("artifacts/udemy_courses.csv")
    print(df.head())  # Verify preprocessing
