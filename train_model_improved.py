"""
Train sentiment model on custom dataset with 70-30 train-test split.
Supports Positive, Negative, Neutral labels.
"""
import os
import sys
import re
import json
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.utils import resample

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Keyword lists for mapping complex sentiment labels
POSITIVE_KEYWORDS = [
    'positive', 'joy', 'excitement', 'contentment', 'happiness', 'love', 'grateful', 'amazing', 
    'excellent', 'great', 'wonderful', 'fantastic', 'happy', 'pleased', 'satisfied', 'delighted', 
    'thrilled', 'ecstatic', 'elated', 'jubilant', 'cheerful', 'optimistic', 'hopeful', 'proud', 
    'triumph', 'heartwarming', 'celebrating', 'victory', 'success', 'achievement', 'gratitude',
    'elation', 'playful', 'serenity', 'bliss', 'euphoria', 'content', 'fulfilled', 'blessed',
    'appreciative', 'thankful', 'inspired', 'motivated', 'energetic', 'enthusiastic', 'passionate',
    'awe', 'pride', 'enthusiasm', 'determination', 'surprise', 'inspiration', 'hope', 'empowerment',
    'admiration', 'compassion', 'tenderness', 'arousal', 'fulfillment', 'reverence', 'thrill',
    'enchantment', 'amusement', 'anticipation', 'kind', 'empathetic', 'free-spirited', 'confident',
    'satisfaction', 'accomplishment', 'harmony', 'creativity', 'wonder', 'adventure', 'affection',
    'adoration', 'zest', 'whimsy', 'radiance', 'rejuvenation', 'resilience', 'exploration',
    'captivation', 'tranquility', 'mischievous', 'motivation', 'appreciation', 'confidence',
    'wonderment', 'optimism', 'intrigue', 'mindfulness', 'elegance', 'melodic', 'innerjourney',
    'freedom', 'dazzle', 'adrenaline', 'artisticburst', 'spark', 'marvel', 'positivity', 'kindness',
    'friendship', 'amazement', 'romance', 'grandeur', 'energy', 'celebration', 'charm', 'ecstasy',
    'colorful', 'connection', 'iconic', 'engagement', 'touched', 'solace', 'breakthrough',
    'vibrancy', 'relief', 'sympathy', 'excited', 'awesome', 'perfect', 'best', 'beautiful',
    'superb', 'outstanding', 'brilliant', 'magnificent', 'fabulous', 'terrific', 'marvelous',
    'lovely', 'nice', 'good', 'like', 'favorite', 'favourite', 'impressive', 'remarkable',
    'pleasant', 'enjoyable', 'delightful', 'satisfying', 'rewarding', 'uplifting', 'inspiring'
]

NEGATIVE_KEYWORDS = [
    'negative', 'sad', 'angry', 'frustrated', 'disappointed', 'terrible', 'awful', 'bad', 'hate', 
    'worst', 'horrible', 'disgusting', 'depressed', 'anxious', 'worried', 'fear', 'stress', 
    'pressure', 'obstacle', 'problem', 'difficulty', 'challenge', 'failure', 'loss', 'pain', 
    'suffering', 'grief', 'sorrow', 'despair', 'hopeless', 'bitterness', 'loneliness', 
    'embarrassed', 'despair', 'hate', 'bitterness', 'resentment', 'rage', 'fury', 'annoyance',
    'irritation', 'disgust', 'contempt', 'shame', 'guilt', 'regret', 'remorse', 'melancholy',
    'gloom', 'misery', 'anguish', 'torment', 'agony', 'distress', 'trouble', 'hardship',
    'anger', 'confusion', 'numbness', 'ambivalence', 'betrayal', 'boredom', 'overwhelmed',
    'desolation', 'bitter', 'jealousy', 'jealous', 'devastated', 'envious', 'dismissive',
    'heartbreak', 'anxiety', 'intimidation', 'helplessness', 'envy', 'yearning', 'apprehensive',
    'isolation', 'disappointment', 'emotionalstorm', 'exhaustion', 'darkness', 'desperation',
    'ruins', 'heartache', 'solitude', 'miscalculation', 'ugly', 'poor', 'dreadful', 'pathetic',
    'atrocious', 'miserable', 'unhappy', 'dislike', 'annoying', 'irritating', 'boring', 'tedious',
    'disappointing', 'frustrating', 'stressful', 'painful', 'tragic', 'miserable', 'upsetting',
    'offensive', 'repulsive', 'revolting', 'abysmal', 'inferior', 'lousy', 'dismal', 'dire'
]

NEUTRAL_KEYWORDS = [
    'neutral', 'okay', 'fine', 'average', 'normal', 'regular', 'standard', 'typical', 'ordinary', 
    'moderate', 'balanced', 'calm', 'indifferent', 'unbiased', 'objective', 'factual', 'informative',
    'curiosity', 'wondering', 'questioning', 'contemplative', 'reflective', 'thoughtful', 'pensive',
    'analytical', 'logical', 'rational', 'practical', 'matter-of-fact',
    'acceptance', 'indifference', 'reflection', 'contemplation', 'emotion', 'journey', 'immersion',
    'nostalgia', 'uncertain', 'unsure', 'mixed', 'ambivalent', 'undecided'
]

def map_sentiment_label(label):
    """Map complex sentiment labels to 0 (Negative), 1 (Neutral), 2 (Positive)."""
    label_str = str(label).strip().lower()
    
    # Direct mapping
    if label_str == "negative":
        return 0
    elif label_str == "neutral":
        return 1
    elif label_str == "positive":
        return 2
    
    # Numeric labels
    try:
        iv = int(label_str)
        if iv in [0, 1, 2]:
            return iv
        if iv in [3, 4, 5]:
            return 2
    except (ValueError, TypeError):
        pass
    
    # Keyword matching
    for kw in POSITIVE_KEYWORDS:
        if kw in label_str:
            return 2
    for kw in NEGATIVE_KEYWORDS:
        if kw in label_str:
            return 0
    for kw in NEUTRAL_KEYWORDS:
        if kw in label_str:
            return 1
    
    # TextBlob fallback
    try:
        tb = TextBlob(label_str)
        if tb.sentiment.polarity > 0.05:
            return 2
        elif tb.sentiment.polarity < -0.05:
            return 0
    except:
        pass
    
    return 1  # Default to Neutral

def clean_text(text):
    """Clean text data."""
    if pd.isna(text) or text == "":
        return ""
    text = str(text)
    if re.match(r'^[\d\s.,+-]+$', text.strip()):
        return ""
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_columns(df):
    """Auto-detect text and label columns."""
    cols = list(df.columns)
    text_col = None
    label_col = None
    
    for col in cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['text', 'comment', 'review', 'content', 'message', 'tweet', 'post']):
            text_col = col
            break
    
    if text_col is None:
        for col in cols:
            if df[col].dtype == 'object' and df[col].astype(str).str.len().mean() > 20:
                text_col = col
                break
    
    for col in cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['sentiment', 'label', 'emotion', 'polarity', 'class']):
            if col != text_col:
                label_col = col
                break
    
    if label_col is None and text_col:
        for col in cols:
            if col != text_col and df[col].nunique() <= 10:
                label_col = col
                break
    
    return text_col, label_col

def main():
    print("="*60)
    print("TRAINING SENTIMENT MODEL (70-30 Split)")
    print("="*60)
    
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "data", "sentimentdataset.csv")
    if not os.path.exists(data_path):
        # Fallback to other common locations
        for alt_path in [
            os.path.join(os.path.dirname(__file__), "sentimentdataset.csv"),
            os.path.join(os.path.dirname(__file__), "IMDB Dataset.csv"),
        ]:
            if os.path.exists(alt_path):
                data_path = alt_path
                break
        else:
            print("ERROR: No dataset found. Place 'sentimentdataset.csv' in data/ folder.")
            sys.exit(1)
    
    print(f"\nLoading dataset: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"Loaded {len(df)} rows")
    
    # Detect columns
    text_col, label_col = detect_columns(df)
    if text_col is None or label_col is None:
        print("ERROR: Could not detect text and label columns.")
        print(f"Columns found: {df.columns.tolist()}")
        sys.exit(1)
    
    print(f"Text column: '{text_col}'")
    print(f"Label column: '{label_col}'")
    
    # Clean text and map labels
    df['cleaned'] = df[text_col].apply(clean_text)
    df['label'] = df[label_col].apply(map_sentiment_label)
    
    # Remove empty texts and unmappable labels
    df = df[(df['cleaned'].str.len() > 5) & (df['label'].isin([0, 1, 2]))].reset_index(drop=True)
    
    print(f"\nValid samples after filtering: {len(df)}")
    label_counts = df['label'].value_counts().sort_index()
    label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for lbl, count in label_counts.items():
        print(f"  {label_names[lbl]}: {count} ({count/len(df)*100:.1f}%)")
    
    if len(df) < 50:
        print("ERROR: Not enough valid samples (need at least 50).")
        sys.exit(1)
    
    # 70-30 train-test split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    
    X_train = train_df['cleaned'].tolist()
    y_train = train_df['label'].tolist()
    X_test = test_df['cleaned'].tolist()
    y_test = test_df['label'].tolist()
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Build and compare models
    print("\nBuilding TF-IDF + Classifier pipeline...")
    
    best_model = None
    best_score = 0
    best_name = ""
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ('classifier', LogisticRegression(
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
            C=1.0
        ))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred, average='macro')
    print(f"LR - Accuracy: {lr_acc*100:.2f}%, Macro F1: {lr_f1*100:.2f}%")
    
    if lr_f1 > best_score:
        best_score = lr_f1
        best_model = lr_pipeline
        best_name = "Logistic Regression"
    
    # 2. XGBoost (only for smaller datasets to avoid memory issues)
    if XGBOOST_AVAILABLE and len(X_train) <= 15000:
        print("\nTraining XGBoost...")
        xgb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )),
            ('classifier', xgb.XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                n_jobs=-1
            ))
        ])
        xgb_pipeline.fit(X_train, y_train)
        xgb_pred = xgb_pipeline.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_f1 = f1_score(y_test, xgb_pred, average='macro')
        print(f"XGB - Accuracy: {xgb_acc*100:.2f}%, Macro F1: {xgb_f1*100:.2f}%")
        
        if xgb_f1 > best_score:
            best_score = xgb_f1
            best_model = xgb_pipeline
            best_name = "XGBoost"
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*60}")
    
    # Final evaluation
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Macro F1: {test_f1*100:.2f}%")
    print(f"Weighted F1: {test_f1_weighted*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print(cm)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metrics
    metrics = {
        "dataset": os.path.basename(data_path),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "split_ratio": "70-30",
        "model": best_name,
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "test_f1_weighted": float(test_f1_weighted),
        "label_distribution": {
            label_names[int(k)]: int(v) for k, v in df['label'].value_counts().items()
        }
    }
    metrics_path = os.path.join(os.path.dirname(__file__), "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
