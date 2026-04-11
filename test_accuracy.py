import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from train_model import map_sentiment_label as tml, clean_text

df = pd.read_csv('d:/Sentiment-Analysis/Sentiment-Analysis-main/data/sentimentdataset.csv')
texts = df['Text'].astype(str).tolist()
cleaned_texts = [clean_text(t) for t in texts]

labels_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
def map_sentiment_label(label):
    iv = tml(label)
    return labels_map[iv] if isinstance(iv, int) else iv

mapped_labels = [map_sentiment_label(str(l)) for l in df['Sentiment'].tolist()]

model = joblib.load('d:/Sentiment-Analysis/Sentiment-Analysis-main/model.joblib')
pred_numeric = model.predict(cleaned_texts)
pred_labels = [labels_map[p] for p in pred_numeric]

acc = accuracy_score(mapped_labels, pred_labels)

with open('result_acc.txt', 'w', encoding='utf-8') as f:
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write("Mapped:\n" + str(pd.Series(mapped_labels).value_counts()) + "\n")
    f.write("Pred:\n" + str(pd.Series(pred_labels).value_counts()) + "\n")
