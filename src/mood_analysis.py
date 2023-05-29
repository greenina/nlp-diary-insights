import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Load the data
df = pd.read_csv('../data/practice.csv')

# Initialize a Lemmatizer
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    text = [lemmatizer.lemmatize(word) for word in text.split(
    ) if word not in stopwords.words('english')]

    text = ' '.join(text)

    return text


# Preprocess diary entries
df['diary_contents'] = df['diary_contents'].apply(preprocess_text)

# Feature Extraction


vectorizer = TfidfVectorizer()

# Fit the vectorizer to the diary entries
features = vectorizer.fit_transform(df['diary_contents'])


# Sentiment Analysis


# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Get the sentiment scores for each diary entry
df['sentiment_scores'] = df['diary_contents'].apply(
    lambda text: sia.polarity_scores(text))

# Get the compound sentiment score for each diary entry
df['compound_score'] = df['sentiment_scores'].apply(
    lambda score_dict: score_dict['compound'])

# Classify each diary entry as 'Positive', 'Neutral', or 'Negative'
df['sentiment'] = df['compound_score'].apply(
    lambda score: 'Positive' if score > 0 else ('Neutral' if score == 0 else 'Negative'))

print(df[:3])
