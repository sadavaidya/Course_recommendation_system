# Course Recommendation System

This project is a content-based course recommendation system that suggests Udemy courses based on user input keywords. The system leverages TF-IDF vectorization and cosine similarity to provide relevant course recommendations.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model and Features](#model-and-features)
- [Running the Streamlit App](#running-the-streamlit-app)
- [License](#license)

## Dataset
The dataset used for this project is sourced from Kaggle: [Udemy Course Dataset](https://www.kaggle.com/datasets).

**Important:** Please check the licensing information on the Kaggle page before using this dataset in production.

## Installation
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/course-recommendation-system.git
cd course-recommendation-system

# Create a virtual environment using Conda
conda create --name course_recommendor python=3.12 -y
conda activate course_recommendor

# Install dependencies
conda install --file requirements.txt
```

## Project Structure
```
├── artifacts/
│   ├── udemy_courses.csv             # Raw dataset
│   ├── cleaned_udemy_courses.csv     # Processed dataset
│   ├── tfidf_vectorizer.pkl          # Saved TF-IDF model
│   ├── similarity_matrix.pkl         # Saved similarity matrix
├── src/
│   ├── data_preprocessing.py         # Cleans and preprocesses data
│   ├── feature_extraction.py         # TF-IDF vectorization
│   ├── recommendation.py             # Course recommendation logic
│   ├── app.py                        # Streamlit web app
├── requirements.txt                   # Required Python libraries
├── environment.yml                    # File to set up environment directly using YAMl file
├── README.md                          # Project documentation

```

## Usage
### Step 1: Data Preprocessing
Run the following command to clean the dataset:
```bash
python src/data_preprocessing.py
```
This script:
- Combines `course_title` and `subject` columns.
- Removes stopwords and special characters.
- Saves the cleaned dataset as `cleaned_udemy_courses.csv`.

### Step 2: Feature Extraction
Extract features using TF-IDF:
```bash
python src/feature_extraction.py
```
This script:
- Converts text data into TF-IDF vectors.
- Saves the `tfidf_vectorizer.pkl` and `similarity_matrix.pkl` files.

### Step 3: Generate Recommendations
Run the recommendation script to test:
```bash
python src/recommendation.py "machine learning"
```
This will return a list of recommended courses based on the keyword provided.

## Running the Streamlit App
To start the web application:
```bash
streamlit run src/app.py
```
This will open a user-friendly interface where users can enter keywords and receive recommended courses.

## License
This project is for educational purposes only. Please review the Kaggle dataset license before using it in production.

---

Feel free to modify the project according to your needs! 🚀

