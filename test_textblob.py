import pandas as pd
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from train_model import map_sentiment_label as tml

df = pd.read_csv('d:/Sentiment-Analysis/Sentiment-Analysis-main/data/sentimentdataset.csv')
texts = df['Text'].astype(str).tolist()

labels_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
def map_sentiment_label(label):
    iv = tml(label)
    return labels_map[iv] if isinstance(iv, int) else iv

mapped_labels = [map_sentiment_label(str(l)) for l in df['Sentiment'].tolist()]

numeric_preds = []
for text in texts:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        numeric_preds.append("Positive")
    elif polarity < -0.05:
        numeric_preds.append("Negative")
    else:
        numeric_preds.append("Neutral")

acc = accuracy_score(mapped_labels, numeric_preds)
print(f"Accuracy with TextBlob: {acc*100:.2f}%")
print(pd.Series(numeric_preds).value_counts())
