import nltk
import pandas as pd
import numpy as np
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Ensure punkt and stopwords are downloaded
nltk.download('punkt')  # Download punkt tokenizer models
nltk.download('stopwords')  # Download stopwords list
nltk.download('punkt_tab')


st.header("Fake News Prediction Project")

# Load your dataset (replace with your dataset path)
df = pd.read_csv(r'C:\Users\admin\Documents\Projects\Fake News Prediction Project\train.csv')

# Check the columns to verify the target variable
st.write(f"Columns in dataset: {df.columns}")

# Preprocessing: Tokenization, stopword removal, and stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Ensure that the text is a valid string, and handle non-string values (e.g., NaN)
    if not isinstance(text, str):
        text = str(text)  # Convert any non-string value to string (like NaN)

    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    
    return " ".join(tokens)

# Apply preprocessing to the dataset
df['processed_text'] = df['text'].apply(preprocess_text)

# Display first few rows of the dataset for better inspection
st.write("First few rows of the dataset:")
st.write(df.head())

# **Important: Inspect your dataset and ensure the column that corresponds to the target label exists**
# Here, we'll assume the target column is called 'label'. Change it if necessary.
target_column = 'label'  # Replace with the correct column name if it's different

# Check if the target column exists
if target_column not in df.columns:
    st.write(f"Error: '{target_column}' column does not exist in the dataset!")
else:
    # Prepare features and labels
    X = df['processed_text']
    y = df[target_column]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


