"""
Robust Sentiment Model Training Pipeline
Trains a high-accuracy sentiment model on any dataset with proper validation and metrics.
Usage:
  python train_custom_model.py --data your_dataset.csv
  python train_custom_model.py --data your_dataset.csv --text-col "Comment" --label-col "Sentiment"
"""
import os
import sys
import re
import argparse
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

# Optional XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False

# Standard label mapping
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
REVERSE_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Extended keyword lists for mapping complex sentiment labels
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
    'nostalgia', 'neutral', 'uncertain', 'unsure', 'mixed', 'ambivalent', 'undecided'
]


def map_sentiment_label(label):
    """
    Map complex sentiment labels to 3 classes: Negative (0), Neutral (1), Positive (2).
    Returns integer labels for model training.
    """
    label_str = str(label).strip().lower()
    
    # Direct mapping if already in standard format
    if label_str in LABEL_MAP:
        return LABEL_MAP[label_str]
    
    # Map numeric labels
    try:
        iv = int(label_str)
        if iv in [0, 1, 2]:
            return iv
        if iv == 4:
            return 2  # In 5-class datasets, 4 is usually Positive
        if iv == 3:
            return 2  # In 5-class datasets, 3 is often Positive
        if iv == 5:
            return 2  # In 5-class datasets, 5 is usually very Positive
    except (ValueError, TypeError):
        pass
    
    # Check for positive keywords
    for keyword in POSITIVE_KEYWORDS:
        if keyword in label_str:
            return 2
    
    # Check for negative keywords
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in label_str:
            return 0
    
    # Check for neutral keywords
    for keyword in NEUTRAL_KEYWORDS:
        if keyword in label_str:
            return 1
    
    # TextBlob fallback
    try:
        tb = TextBlob(label_str)
        if tb.sentiment.polarity > 0.05:
            return 2
        elif tb.sentiment.polarity < -0.05:
            return 0
    except Exception:
        pass
    
    return 1  # Default to Neutral


def clean_text(text):
    """Clean and preprocess text data."""
    if pd.isna(text) or text == "":
        return ""
    text = str(text)
    # Remove purely numeric text
    if re.match(r'^[\d\s.,+-]+$', text.strip()):
        return ""
    # Remove special characters but keep spaces, letters, numbers, and basic punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_columns(df):
    """Auto-detect text and label columns in the dataset."""
    cols = list(df.columns)
    text_col = None
    label_col = None
    
    # Find text column
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
    
    # Find label column
    for col in cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['sentiment', 'label', 'emotion', 'polarity', 'class']):
            if col != text_col:
                label_col = col
                break
    
    if label_col is None and text_col:
        # Try to find a column with few unique values (likely labels)
        for col in cols:
            if col != text_col and df[col].nunique() <= 10:
                label_col = col
                break
    
    return text_col, label_col


def load_and_prepare_data(data_path, text_col=None, label_col=None):
    """Load dataset and prepare text and labels."""
    # Read CSV
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='replace')
    
    # Auto-detect columns if not provided
    if text_col is None or label_col is None:
        detected_text, detected_label = detect_columns(df)
        if text_col is None:
            text_col = detected_text
        if label_col is None:
            label_col = detected_label
    
    if text_col is None:
        raise ValueError("Could not detect text column. Please specify --text-col")
    if label_col is None:
        raise ValueError("Could not detect label column. Please specify --label-col")
    
    print(f"Using text column: '{text_col}'")
    print(f"Using label column: '{label_col}'")
    
    # Clean texts
    texts = df[text_col].apply(clean_text).tolist()
    
    # Map labels
    labels = df[label_col].apply(map_sentiment_label).tolist()
    
    # Filter valid data
    valid_indices = [i for i, (t, l) in enumerate(zip(texts, labels)) 
                     if t and len(t.strip()) > 0 and l in [0, 1, 2]]
    
    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    print(f"\nDataset Statistics:")
    print(f"  Total rows: {len(df)}")
    print(f"  Valid samples after filtering: {len(texts)}")
    
    label_counts = pd.Series(labels).value_counts().sort_index()
    for label_idx, count in label_counts.items():
        print(f"  {REVERSE_LABEL_MAP[label_idx]}: {count} ({count/len(texts)*100:.1f}%)")
    
    return texts, labels


def upsample_minority(X, y):
    """Upsample minority classes to balance dataset."""
    df = pd.DataFrame({"text": X, "label": y})
    counts = df['label'].value_counts()
    max_count = counts.max()
    frames = []
    for lbl in counts.index:
        subset = df[df['label'] == lbl]
        if len(subset) < max_count:
            subset_up = resample(subset, replace=True, n_samples=max_count, random_state=42)
            frames.append(subset_up)
        else:
            frames.append(subset)
    new_df = pd.concat(frames).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return new_df['text'].tolist(), new_df['label'].tolist()


def build_pipeline(use_xgboost=True):
    """Build TF-IDF + Classifier pipeline."""
    if use_xgboost and XGBOOST_AVAILABLE:
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            n_jobs=-1,
            random_state=42,
            eval_metric='mlogloss'
        )
    else:
        clf = LogisticRegression(
            max_iter=2000,
            random_state=42,
            solver='lbfgs'
        )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95,
            lowercase=True,
            strip_accents='unicode'
        )),
        ('classifier', clf)
    ])
    return pipeline


def train_model(texts, labels, use_xgboost=True, n_iter=50, upsample=True):
    """Train model with hyperparameter tuning and validation. Tries both LR and XGBoost, returns best."""
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    print(f"\nSplit: Train={len(X_train)}, Validation={len(X_val)}")
    
    best_overall_model = None
    best_overall_score = 0
    best_method_name = ""
    
    # Try Logistic Regression
    print("\n" + "="*40)
    print("Training Logistic Regression...")
    print("="*40)
    
    lr_train, lr_y_train = (upsample_minority(X_train, y_train) if upsample else (X_train[:], y_train[:]))
    
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english', min_df=2, max_df=0.95)),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs'))
    ])
    
    lr_params = {
        'tfidf__max_features': [10000, 15000, 20000, 25000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'classifier__C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    lr_search = RandomizedSearchCV(lr_pipeline, lr_params, n_iter=min(n_iter, 30), cv=cv, scoring='f1_macro', verbose=0, n_jobs=-1, random_state=42)
    lr_search.fit(lr_train, lr_y_train)
    
    lr_model = lr_search.best_estimator_
    lr_pred = lr_model.predict(X_val)
    lr_score = f1_score(y_val, lr_pred, average='macro')
    lr_acc = accuracy_score(y_val, lr_pred)
    print(f"LR - Accuracy: {lr_acc*100:.2f}%, Macro F1: {lr_score*100:.2f}%")
    
    if lr_score > best_overall_score:
        best_overall_score = lr_score
        best_overall_model = lr_model
        best_method_name = "Logistic Regression"
    
    # Try XGBoost if available
    if use_xgboost and XGBOOST_AVAILABLE:
        print("\n" + "="*40)
        print("Training XGBoost...")
        print("="*40)
        
        xgb_train, xgb_y_train = (upsample_minority(X_train, y_train) if upsample else (X_train[:], y_train[:]))
        
        xgb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english', min_df=2, max_df=0.95)),
            ('classifier', xgb.XGBClassifier(objective="multi:softprob", n_jobs=-1, random_state=42, eval_metric='mlogloss'))
        ])
        
        xgb_params = {
            'tfidf__max_features': [10000, 15000, 20000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__n_estimators': randint(100, 400),
            'classifier__max_depth': randint(3, 8),
            'classifier__learning_rate': uniform(0.05, 0.2),
            'classifier__subsample': uniform(0.7, 0.3),
            'classifier__reg_alpha': uniform(0.0, 0.5),
            'classifier__reg_lambda': uniform(0.5, 1.5),
        }
        
        xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_params, n_iter=min(n_iter, 30), cv=cv, scoring='f1_macro', verbose=0, n_jobs=-1, random_state=42)
        xgb_search.fit(xgb_train, xgb_y_train)
        
        xgb_model = xgb_search.best_estimator_
        xgb_pred = xgb_model.predict(X_val)
        xgb_score = f1_score(y_val, xgb_pred, average='macro')
        xgb_acc = accuracy_score(y_val, xgb_pred)
        print(f"XGB - Accuracy: {xgb_acc*100:.2f}%, Macro F1: {xgb_score*100:.2f}%")
        
        if xgb_score > best_overall_score:
            best_overall_score = xgb_score
            best_overall_model = xgb_model
            best_method_name = "XGBoost"
    
    print(f"\nBest method: {best_method_name} (Macro F1: {best_overall_score*100:.2f}%)")
    
    best_model = best_overall_model
    
    # Final evaluation
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({best_method_name})")
    print(f"{'='*60}")
    print(f"Accuracy: {val_acc*100:.2f}%")
    print(f"Macro F1: {val_f1*100:.2f}%")
    print(f"Weighted F1: {val_f1_weighted*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=["Negative", "Neutral", "Positive"]))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    
    return best_model, val_acc, val_f1, val_f1_weighted


def evaluate_on_test_data(model, texts, labels, dataset_name="Test"):
    """Evaluate model on a separate test dataset."""
    y_pred = model.predict(texts)
    acc = accuracy_score(labels, y_pred)
    f1 = f1_score(labels, y_pred, average='weighted')
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} DATASET EVALUATION")
    print(f"{'='*60}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Weighted F1: {f1*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(labels, y_pred, target_names=["Negative", "Neutral", "Positive"]))
    
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--data", required=True, help="CSV file path for training")
    parser.add_argument("--text-col", default=None, help="Text column name (auto-detected if not specified)")
    parser.add_argument("--label-col", default=None, help="Label column name (auto-detected if not specified)")
    parser.add_argument("--output", default="model.joblib", help="Output model path")
    parser.add_argument("--use-xgb", action="store_true", help="Use XGBoost classifier")
    parser.add_argument("--n-iter", type=int, default=50, help="Hyperparameter search iterations")
    parser.add_argument("--no-upsample", action="store_true", help="Disable upsampling")
    parser.add_argument("--test-data", default=None, help="Optional test CSV to evaluate generalization")
    parser.add_argument("--test-text-col", default=None, help="Test data text column")
    parser.add_argument("--test-label-col", default=None, help="Test data label column")
    args = parser.parse_args()
    
    print("="*60)
    print("SENTIMENT MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load training data
    print(f"\nLoading training data: {args.data}")
    texts, labels = load_and_prepare_data(args.data, args.text_col, args.label_col)
    
    if len(texts) < 50:
        print("ERROR: Not enough valid samples. Need at least 50.")
        sys.exit(1)
    
    # Train model
    use_xgb = args.use_xgb or XGBOOST_AVAILABLE
    upsample = not args.no_upsample
    
    best_model, val_acc, val_f1, val_f1_weighted = train_model(
        texts, labels, 
        use_xgboost=use_xgb, 
        n_iter=args.n_iter,
        upsample=upsample
    )
    
    # Evaluate on test data if provided
    if args.test_data:
        print(f"\n\nLoading test data: {args.test_data}")
        test_texts, test_labels = load_and_prepare_data(
            args.test_data, 
            args.test_text_col, 
            args.test_label_col
        )
        evaluate_on_test_data(best_model, test_texts, test_labels, "External Test")
    
    # Save model
    joblib.dump(best_model, args.output)
    print(f"\n{'='*60}")
    print(f"Model saved to: {args.output}")
    print(f"{'='*60}")
    
    # Save training metrics
    metrics = {
        "validation_accuracy": float(val_acc),
        "validation_f1_macro": float(val_f1),
        "validation_f1_weighted": float(val_f1_weighted),
        "training_samples": len(texts),
        "label_distribution": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()}
    }
    metrics_path = args.output.replace('.joblib', '_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    return best_model


if __name__ == "__main__":
    main()
