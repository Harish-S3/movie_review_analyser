# Moview Review based  on Sentiment Analysis 

A comprehensive sentiment analysis pipeline that combines text preprocessing, feature engineering, and machine learning to classify movie reviews with high accuracy.

## 📊 Project Overview

This project implements an optimized sentiment analysis system that processes IMDB movie reviews and classifies them as positive or negative. The pipeline includes advanced text preprocessing, parallel processing, feature engineering with TF-IDF and sentiment scores, and a logistic regression model for classification.

## 🚀 Features

### Advanced Text Processing
- **HTML Cleaning**: Removes HTML tags from reviews
- **Parallel Processing**: Utilizes multi-core processing for faster text preprocessing
- **Custom Stopwords**: Enhanced stopword list with movie-specific terms
- **Lemmatization**: Reduces words to their base forms using WordNet lemmatizer

### Feature Engineering
- **TF-IDF Vectorization**: With unigrams and bigrams (8,000 features)
- **Sentiment Scoring**: Uses TextBlob for additional sentiment features
- **Feature Combination**: Combines TF-IDF features with sentiment scores

### Model Training
- **Logistic Regression**: Optimized with saga solver and L2 regularization
- **Multi-core Training**: Utilizes all available CPU cores
- **Comprehensive Evaluation**: Accuracy score and classification report

### Visualization
- **Sentiment Visualization**: Color-coded word importance display
- **Polarity Scoring**: Individual word and overall review sentiment analysis

## 📁 Project Structure

```
sentiment-analysis/
├── imdb_reviews.csv          # Dataset (not included in repo)
├── preprocess_cache/         # Cache directory for processed data
│   └── preprocessed_reviews.joblib
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛠️ Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd sentiment-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK resources** (automatically handled by the script)
```python
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')
```

4. **Place your dataset**
- Ensure `imdb_reviews.csv` is in the project root
- Expected columns: `review` (text), `sentiment` (label)

## 📊 Dataset Format

The script expects a CSV file with the following structure:

| review | sentiment |
|--------|-----------|
| "This movie was amazing..." | positive |
| "Terrible acting and plot..." | negative |

## 🎯 Usage

Run the main script:

```bash
python sentiment_analysis.py
```

The script will:
1. Load and preprocess the data
2. Train the sentiment classification model
3. Evaluate performance
4. Display a sample sentiment visualization

## ⚡ Performance Optimizations

- **Parallel Processing**: Text preprocessing using all available CPU cores
- **Caching**: Preprocessed data is cached to disk for faster subsequent runs
- **Efficient Vectorization**: TF-IDF with optimized parameters
- **Multi-core Model Training**: Logistic regression trained using all cores

## 📈 Model Performance

The optimized pipeline achieves:
- High accuracy on IMDB sentiment classification
- Detailed classification report with precision, recall, and F1-score
- Efficient processing of large text datasets

## 🎨 Sample Output

The script includes a creative sentiment visualization that shows:
- Color-coded words (green=positive, red=negative, default=neutral)
- Word importance indicators
- Overall polarity score
- Predicted sentiment

Example:
```
==================================================
Sentiment Visualization:
==================================================
Word Impact (size indicates importance in this review):
amazing!!! great!!! performance!! 
--------------------------------------------------
Overall Polarity: 0.8750
Predicted Sentiment: positive
==================================================
```

## 🔧 Customization

### Adjusting Parameters
- Modify `max_features` in `TfidfVectorizer` for different vocabulary sizes
- Change `ngram_range` for different n-gram combinations
- Adjust `C` parameter in `LogisticRegression` for regularization strength

### Adding New Features
- Extend the feature engineering section to include additional text features
- Incorporate other sentiment analysis libraries (VADER, etc.)
- Add metadata features if available in your dataset

## 📋 Dependencies

- pandas
- numpy
- scikit-learn
- nltk
- textblob
- beautifulsoup4
- joblib
- scipy

## 🎓 Applications

This sentiment analysis pipeline can be adapted for:
- Product review analysis
- Social media monitoring
- Customer feedback processing
- Market research
- Content moderation

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check issues page.

## 📞 Support

If you have any questions or need help with implementation, please open an issue in the repository.

---

**Note**: The dataset `imdb_reviews.csv` is not included in this repository. You'll need to provide your own dataset in the specified format.
