import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load dataset
df = pd.read_csv('IMDB Dataset.csv')
# Data cleaning and preprocessing
df['review'] = df['review'].str.replace('<br />', ' ').str.replace('[^a-zA-Z ]', '').str.lower()
print(df)



def get_sentiment(review):
    headers = {
        'Authorization': 'Bearer API_KEY',
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'user', 'content': f'Classify the sentiment of this review: "{review}"'}],
        'max_tokens': 60,
    }

    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    return response.json()['choices'][0]['message']['content'].strip()


results = []
for review in df['review']:
    sentiment = get_sentiment(review)
    results.append(sentiment)
df['predicted_sentiment'] = results



accuracy = accuracy_score(df['actual_sentiment'], df['predicted_sentiment'])
precision = precision_score(df['actual_sentiment'], df['predicted_sentiment'], pos_label='positive')
recall = recall_score(df['actual_sentiment'], df['predicted_sentiment'], pos_label='positive')
f1 = f1_score(df['actual_sentiment'], df['predicted_sentiment'], pos_label='positive')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
