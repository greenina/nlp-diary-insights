import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from collections import Counter
from itertools import islice


raw_df = pd.read_csv('../data/dataset.csv')
raw_df['diary_contents'] = raw_df['diary_contents'].replace(
    np.nan, '', regex=True)

raw_df['diary_contents'] = raw_df['diary_contents'].astype(str)

df = raw_df.groupby('written_date')[
    'diary_contents'].apply(' '.join).reset_index()

df['written_date'] = pd.to_datetime(df['written_date'], format='%A, %B %d, %Y')
df = df.sort_values('written_date')
df = df.reset_index(drop=True)

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

sia = SentimentIntensityAnalyzer()

df['sentiment_scores'] = df['diary_contents'].apply(
    lambda text: sia.polarity_scores(text))

df['compound_score'] = df['sentiment_scores'].apply(
    lambda score_dict: score_dict['compound'])

df['sentiment'] = df['compound_score'].apply(
    lambda score: 'Positive' if score > 0 else ('Neutral' if score == 0 else 'Negative'))


nrc_lexicon = pd.read_csv('../data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                          names=["word", "emotion", "association"], sep='\t')

nrc_lexicon = nrc_lexicon.pivot(
    index='word', columns='emotion', values='association').reset_index()

df['diary_contents'] = df['diary_contents'].apply(lambda x: x.split())


def get_emotion_count(words):
    words_in_lexicon = nrc_lexicon[nrc_lexicon['word'].isin(words)]

    total_words = len(words)

    emotion_count = (words_in_lexicon.drop(
        'word', axis=1).sum() / total_words).to_dict()

    return emotion_count


df['emotion_count'] = df['diary_contents'].apply(get_emotion_count)


# plt.plot(df['written_date'], df['compound_score'])
# plt.title('Mood Over Time')
# plt.xlabel('Date')
# plt.ylabel('Mood')
# plt.show()


emotion_sums = df.groupby('written_date').apply(lambda x: pd.Series({
    emotion: sum(entry.get(emotion, 0) for entry in x['emotion_count'])
    for emotion in ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust', 'positive', 'negative']
}))

emotion_sums.plot(kind='line', subplots=True, layout=(
    5, 2), sharex=True, figsize=(10, 10), title='Mood Over Time')
plt.show()

emotion_sequences = emotion_sums.apply(
    lambda x: [emotion for emotion, count in x.items() for _ in range(int(count))], axis=1)

emotion_pairs = [pair for sequence in emotion_sequences for pair in zip(
    sequence, islice(sequence, 1, None))]

pair_counts = Counter(emotion_pairs)

most_common_pairs = pair_counts.most_common(10)

total_pairs = sum(pair_counts.values())

for pair, count in most_common_pairs:
    percentage = count / total_pairs * 100
    print(
        f"When one feels {pair[0]}, it's {percentage:.2f}% likely that one will feel {pair[1]} the next day.")

historical_events = {
    "1942-06-12": "Anne Frank's 13th birthday",
    "1942-07-06": "Anne Frank and her family go into hiding in the Secret Annex in Amsterdam, Netherlands",
    "1942-08-01": "Anne starts writing about other Jews hiding with her family in the Secret Annex",
    "1944-08-04": "The Secret Annex is discovered by the Nazis, Anne Frank and her family are arrested",
    "1942-01-20": "Wannsee Conference: The Nazis decide on the 'Final Solution' for Europe's Jews",
    "1942-06-01": "Operation Blau: The German army attacks the Soviet Union, capturing several cities including Stalingrad",
    "1943-04-19": "Warsaw Ghetto Uprising: Jews in the Warsaw Ghetto rise up against the Nazis",
    "1944-06-06": "D-Day: The Allied forces land in Normandy, France, beginning the end of Nazi Germany",
    "1944-07-20": "Operation Valkyrie: Failed attempt to assassinate Hitler within the German army"
}

df['historical_event'] = df['written_date'].map(historical_events)


for event_date, event_description in historical_events.items():
    event_date = pd.to_datetime(event_date)
    start_date = event_date - pd.Timedelta(days=7)
    end_date = event_date + pd.Timedelta(days=7)
    df_event = df[(df['written_date'] >= start_date)
                  & (df['written_date'] <= end_date)]
    plt.figure(figsize=(10, 5))
    plt.plot(df_event['written_date'], df_event['compound_score'])
    plt.title(event_description)
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.show()
