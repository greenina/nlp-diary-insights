import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('../data/practice.csv')

lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    text = text.lower()

    text = ''.join([char for char in text if char not in string.punctuation])

    text = [lemmatizer.lemmatize(word) for word in text.split(
    ) if word not in stopwords.words('english')]

    text = ' '.join(text)

    return text


df['diary_contents'] = df['diary_contents'].apply(preprocess_text)


nrc_lexicon = pd.read_csv('../data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                          names=["word", "emotion", "association"], sep='\t')

nrc_lexicon = nrc_lexicon.pivot(
    index='word', columns='emotion', values='association').reset_index()

df['diary_contents'] = df['diary_contents'].apply(lambda x: x.split())


def get_emotion_count(words):
    words_in_lexicon = nrc_lexicon[nrc_lexicon['word'].isin(words)]

    emotion_count = words_in_lexicon.drop('word', axis=1).sum().to_dict()

    return emotion_count


df['emotion_count'] = df['diary_contents'].apply(get_emotion_count)

df['written_date'] = pd.to_datetime(df['written_date'], format='%A, %B %d, %Y')

emotion_sums = df.groupby('written_date').apply(lambda x: pd.Series({
    emotion: sum(entry.get(emotion, 0) for entry in x['emotion_count'])
    for emotion in ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust', 'positive', 'negative']
}))

emotion_sums.plot(kind='line', subplots=True, layout=(
    5, 2), sharex=True, figsize=(10, 10), title='Mood Over Time')
plt.show()
