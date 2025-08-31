import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from joblib import Parallel, delayed
import os
import time

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load data
print("Loading dataset...")
data = pd.read_csv('imdb_reviews.csv')
print(f"Loaded {len(data)} reviews")

# 1. HTML Cleaning
print("Cleaning HTML tags...")
def remove_html(text):
    return BeautifulSoup(text, 'html.parser').get_text()

data['clean_review'] = data['review'].apply(remove_html)

# 2. Optimized Preprocessing
stop_words = set(stopwords.words('english'))
custom_stopwords = {'movie', 'film', 'br', 'one', 'make', 'like', 'even'}
stop_words = stop_words.union(custom_stopwords)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenization and cleaning
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Parallel processing for faster preprocessing
print("Preprocessing reviews (parallel processing)...")
start_time = time.time()

# Create cache directory if not exists
cache_dir = "preprocess_cache"
os.makedirs(cache_dir, exist_ok=True)
cache_file = os.path.join(cache_dir, "preprocessed_reviews.joblib")

# Check if cached version exists
if os.path.exists(cache_file):
    print("Loading preprocessed data from cache...")
    data = pd.read_pickle(cache_file)
else:
    # Process in parallel
    num_cores = os.cpu_count() - 1 or 1
    results = Parallel(n_jobs=num_cores)(
        delayed(preprocess_text)(review) 
        for review in data['clean_review']
    )
    data['processed_review'] = results
    data.to_pickle(cache_file)
    print(f"Preprocessed data cached to {cache_file}")

print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

# 3. Sentiment Feature Engineering
print("Calculating sentiment scores...")
data['sentiment_score'] = data['clean_review'].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)

# 4. Split data
print("Splitting data...")
X = data[['processed_review', 'sentiment_score']]
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Feature Engineering with TF-IDF
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Unigrams and bigrams
    max_features=8000,
    stop_words=list(stop_words)
)
X_train_tfidf = vectorizer.fit_transform(X_train['processed_review'])
X_test_tfidf = vectorizer.transform(X_test['processed_review'])

# Combine TF-IDF with sentiment scores
print("Combining features...")
X_train_combined = hstack([
    X_train_tfidf, 
    csr_matrix(X_train['sentiment_score'].values.reshape(-1, 1))
])
X_test_combined = hstack([
    X_test_tfidf, 
    csr_matrix(X_test['sentiment_score'].values.reshape(-1, 1))
])

# 6. Efficient Model Training
print("Training model...")
model = LogisticRegression(
    max_iter=1000, 
    C=10, 
    solver='saga', 
    n_jobs=-1  # Use all cores
)
model.fit(X_train_combined, y_train)

# 7. Prediction and Evaluation
print("Evaluating model...")
predictions = model.predict(X_test_combined)
print(f"Optimized Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions))

# 8. Fixed Creative Output: Sentiment Visualization
def visualize_sentiment(review):
    """Create a sentiment visualization for a review"""
    # Preprocess the text
    processed_text = preprocess_text(review)
    
    # Calculate sentiment score
    sentiment_val = TextBlob(review).sentiment.polarity
    
    # Transform text to TF-IDF features
    tfidf_features = vectorizer.transform([processed_text])
    
    # Combine with sentiment score
    combined_features = hstack([
        tfidf_features, 
        csr_matrix([[sentiment_val]])
    ])
    
    # Get TF-IDF scores for words
    words = processed_text.split()
    word_scores = []
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        if clean_word in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[clean_word]
            score = tfidf_features[0, idx]  # Use current review's TF-IDF score
            word_scores.append((word, score))
        else:
            word_scores.append((word, 0))
    
    # Create visualization
    print("\n" + "="*50)
    print("Sentiment Visualization:")
    print("="*50)
    print("Word Impact (size indicates importance in this review):")
    
    for word, score in word_scores:
        # Scale the score for display
        size = min(int(abs(score) * 100) + 1, 5)
        # Get word polarity
        word_polarity = TextBlob(word).sentiment.polarity
        
        # Add color based on sentiment
        if word_polarity > 0.1:
            color_code = "\033[92m"  # Green (positive)
        elif word_polarity < -0.1:
            color_code = "\033[91m"  # Red (negative)
        else:
            color_code = "\033[0m"   # Default (neutral)
        
        # Create visual effect - using repetition for "size"
        sized_word = f"{color_code}{word}{'!'*size}\033[0m"
        print(sized_word, end=' ')
    
    print("\n\n" + "-"*50)
    print(f"Overall Polarity: {sentiment_val:.4f}")
    print(f"Predicted Sentiment: {model.predict(combined_features)[0]}")
    print("="*50)

# Sample review analysis
sample = data.iloc[0]['clean_review']
print("\nAnalyzing sample review:")
visualize_sentiment(sample)