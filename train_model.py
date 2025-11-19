"""
Training script for Sentiment Analysis Model
Improved: randomized hyperparameter search, stratified CV, early stopping, upsampling option,
and CLI arguments for reproducible experiments.
Additionally: optional transformer fine-tuning (calls train_transformer.py) and saves a
lightweight transformer wrapper into model.joblib so the app can detect/use it.
"""
import os
import re
import sys
import subprocess
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

# Optional XGBoost import (if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    XGBOOST_AVAILABLE = False

# Label mapping: 0=Negative, 1=Neutral, 2=Positive
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
REVERSE_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Track last detected column names so we can pass them to transformer trainer
LAST_DATA_TEXT_COLUMN = None
LAST_DATA_LABEL_COLUMN = None

# Extended keyword lists for mapping complex sentiment labels to 3 classes
POSITIVE_KEYWORDS = ['positive', 'joy', 'excitement', 'contentment', 'happiness', 'love', 'grateful', 'amazing', 
                     'excellent', 'great', 'wonderful', 'fantastic', 'happy', 'pleased', 'satisfied', 'delighted', 
                     'thrilled', 'ecstatic', 'elated', 'jubilant', 'cheerful', 'optimistic', 'hopeful', 'proud', 
                     'triumph', 'heartwarming', 'celebrating', 'victory', 'success', 'achievement', 'gratitude',
                     'elation', 'playful', 'serenity', 'bliss', 'euphoria', 'content', 'fulfilled', 'blessed',
                     'appreciative', 'thankful', 'inspired', 'motivated', 'energetic', 'enthusiastic', 'passionate']

NEGATIVE_KEYWORDS = ['negative', 'sad', 'angry', 'frustrated', 'disappointed', 'terrible', 'awful', 'bad', 'hate', 
                     'worst', 'horrible', 'disgusting', 'depressed', 'anxious', 'worried', 'fear', 'stress', 
                     'pressure', 'obstacle', 'problem', 'difficulty', 'challenge', 'failure', 'loss', 'pain', 
                     'suffering', 'grief', 'sorrow', 'despair', 'hopeless', 'bitterness', 'loneliness', 
                     'embarrassed', 'despair', 'hate', 'bitterness', 'resentment', 'rage', 'fury', 'annoyance',
                     'irritation', 'disgust', 'contempt', 'shame', 'guilt', 'regret', 'remorse', 'melancholy',
                     'gloom', 'misery', 'anguish', 'torment', 'agony', 'distress', 'trouble', 'hardship']

NEUTRAL_KEYWORDS = ['neutral', 'okay', 'fine', 'average', 'normal', 'regular', 'standard', 'typical', 'ordinary', 
                    'moderate', 'balanced', 'calm', 'indifferent', 'unbiased', 'objective', 'factual', 'informative',
                    'curiosity', 'wondering', 'questioning', 'contemplative', 'reflective', 'thoughtful', 'pensive',
                    'contemplative', 'analytical', 'logical', 'rational', 'practical', 'matter-of-fact']


def map_sentiment_label(label):
    """
    Map complex sentiment labels to 3 classes: Negative (0), Neutral (1), Positive (2).
    Uses keyword matching to categorize labels.
    """
    label_str = str(label).strip().lower()
    
    # Direct mapping if already in standard format
    if label_str in LABEL_MAP:
        return LABEL_MAP[label_str]
    
    # Check for positive keywords
    for keyword in POSITIVE_KEYWORDS:
        if keyword in label_str:
            return 2  # Positive
    
    # Check for negative keywords
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in label_str:
            return 0  # Negative
    
    # Check for neutral keywords
    for keyword in NEUTRAL_KEYWORDS:
        if keyword in label_str:
            return 1  # Neutral
    
    # Try to parse as integer (0, 1, or 2)
    try:
        iv = int(label_str)
        if iv in [0, 1, 2]:
            return iv
    except (ValueError, TypeError):
        pass
    
    # Default to neutral if unclear
    return 1  # Neutral


def clean_text(text):
    """Clean and preprocess text data."""
    if pd.isna(text) or text == "":
        return ""
    text = str(text)
    # Remove special characters but keep spaces, letters, numbers, and basic punctuation
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(data_path=None):
    """
    Load training data from CSV file or from Hugging Face dataset.
    Returns: texts, labels, test_texts, test_labels (test_* may be None if using CSV without test split)
    """
    global LAST_DATA_TEXT_COLUMN, LAST_DATA_LABEL_COLUMN
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        text_col = None
        label_col = None
        for col in df.columns:
            if 'text' in col.lower() or 'comment' in col.lower() or 'review' in col.lower():
                text_col = col
            if 'sentiment' in col.lower() or 'label' in col.lower():
                label_col = col
        if text_col is None:
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
        if label_col is None:
            raise ValueError("Could not find label column. Expected column with 'sentiment' or 'label' in name.")
        LAST_DATA_TEXT_COLUMN = text_col
        LAST_DATA_LABEL_COLUMN = label_col
        texts = df[text_col].apply(clean_text).tolist()
        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) > 0]
        texts = [texts[i] for i in valid_indices]
        
        # Normalize labels to 0/1/2 using improved mapping function
        labels = df[label_col].apply(map_sentiment_label).tolist()
        labels = [labels[i] for i in valid_indices]  # Match filtered texts
        
        print(f"\nLabel distribution after mapping:")
        label_counts = pd.Series(labels).value_counts().sort_index()
        for label_idx, count in label_counts.items():
            print(f"  {REVERSE_LABEL_MAP[label_idx]}: {count}")
        
        return texts, labels, None, None

    # fallback: Hugging Face tweet_eval
    LAST_DATA_TEXT_COLUMN = None
    LAST_DATA_LABEL_COLUMN = None
    from datasets import load_dataset
    print("No data file provided. Loading Hugging Face 'tweet_eval' dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")
    # convert Column objects to plain lists before concatenation
    train_texts = list(dataset["train"]["text"]) + list(dataset["validation"]["text"])
    train_labels = list(dataset["train"]["label"]) + list(dataset["validation"]["label"])
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])
    train_texts = [clean_text(t) for t in train_texts]
    test_texts = [clean_text(t) for t in test_texts]
    return train_texts, train_labels, test_texts, test_labels


def upsample_minority(X, y):
    """Simple upsampling of minority classes to balance dataset."""
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


def build_pipeline(use_xgboost=True, xgb_params=None):
    """Return a sklearn Pipeline with TF-IDF + classifier."""
    if use_xgboost and XGBOOST_AVAILABLE:
        clf = xgb.XGBClassifier(
            use_label_encoder=False,
            objective="multi:softprob",
            n_jobs=-1,
            random_state=42,
            **(xgb_params or {})
        )
    else:
        clf = LogisticRegression(
            max_iter=2000,
            random_state=42,
            multi_class='multinomial',
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


def tune_and_train(X_train, y_train, X_val, y_val, use_xgboost=True, n_iter=50, upsample=True, n_jobs=1):
    """Hyperparameter tuning with RandomizedSearchCV and final training with early stopping for XGBoost."""
    # Always upsample for imbalanced datasets to improve accuracy
    if upsample:
        print("\nUpsampling minority classes to balance dataset...")
        X_train, y_train = upsample_minority(X_train, y_train)
        print(f"After upsampling - Train size: {len(X_train)}")
        label_counts = pd.Series(y_train).value_counts().sort_index()
        for label_idx, count in label_counts.items():
            print(f"  {REVERSE_LABEL_MAP[label_idx]}: {count}")

    pipeline = build_pipeline(use_xgboost=use_xgboost)

    if use_xgboost and XGBOOST_AVAILABLE:
        param_dist = {
            'tfidf__max_features': [10000, 15000, 20000, 25000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'classifier__n_estimators': randint(200, 800),
            'classifier__max_depth': randint(4, 12),
            'classifier__learning_rate': uniform(0.01, 0.15),
            'classifier__subsample': uniform(0.7, 0.3),
            'classifier__colsample_bytree': uniform(0.7, 0.3),
            'classifier__min_child_weight': randint(1, 5),
            'classifier__gamma': uniform(0.0, 0.3),
            'classifier__reg_alpha': uniform(0.0, 0.5),
            'classifier__reg_lambda': uniform(0.5, 1.0),
        }
    else:
        param_dist = {
            'tfidf__max_features': [8000, 12000, 15000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': uniform(0.1, 10.0),
        }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=n_iter, cv=cv, scoring='f1_macro',
        verbose=2, n_jobs=n_jobs, random_state=42
    )
    print("Starting hyperparameter search...")
    search.fit(X_train, y_train)

    print("\nBest params:")
    print(search.best_params_)
    best = search.best_estimator_

    # If XGBoost, refit with early stopping on validation set
    if use_xgboost and XGBOOST_AVAILABLE:
        clf = best.named_steps['classifier']
        tfidf = best.named_steps['tfidf']
        # Refit pipeline manually to enable early stopping
        X_train_t = tfidf.fit_transform(X_train)
        X_val_t = tfidf.transform(X_val)
        print("\nRefitting XGBoost with early stopping on validation set...")
        clf.set_params(n_jobs=-1, random_state=42)
        # Use eval_set parameter (newer XGBoost API)
        try:
            # Try new API first (XGBoost 2.0+)
            clf.fit(
                X_train_t, y_train,
                eval_set=[(X_val_t, y_val)],
                verbose=False
            )
        except TypeError:
            # Fallback for older XGBoost versions
            clf.fit(X_train_t, y_train)
        # attach refit steps back
        from sklearn.pipeline import Pipeline as SkPipeline
        best = SkPipeline([('tfidf', tfidf), ('classifier', clf)])

    # Final evaluation on validation
    y_val_pred = best.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"\nValidation Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")
    print("\nClassification Report (val):")
    print(classification_report(y_val, y_val_pred, target_names=["Negative", "Neutral", "Positive"]))
    print("\nConfusion Matrix (val):")
    print(confusion_matrix(y_val, y_val_pred))

    return best


class TransformerWrapper:
    """
    Lightweight wrapper referencing a fine-tuned transformer directory.
    The wrapper is pickleable: only the path is serialized. Model/tokenizer are loaded lazily at runtime.
    Methods:
      - predict(texts): returns label ints
      - predict_proba(texts): returns probability arrays (softmax)
    """
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            # Import lazily to avoid requiring transformers at joblib-dump time
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except Exception as e:
            raise RuntimeError("transformers/torch required to use TransformerWrapper: " + str(e))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def predict(self, texts, batch_size=32):
        probs = self.predict_proba(texts, batch_size=batch_size)
        return np.argmax(probs, axis=-1).tolist()

    def predict_proba(self, texts, batch_size=32):
        self._ensure_loaded()
        import torch
        from torch.nn.functional import softmax
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self._tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=128)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._model(**enc)
                logits = out.logits
                probs = softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)


def call_transformer_trainer(transformer_args, python_exe=None):
    """Call train_transformer.py as a subprocess. transformer_args is list of args (e.g. ['--data', 'my.csv'])"""
    python_exe = python_exe or sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "train_transformer.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"train_transformer.py not found at {script_path}")
    cmd = [python_exe, script_path] + transformer_args
    print("Running transformer trainer:", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"train_transformer.py exited with code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--data", help="CSV file path; if omitted uses Hugging Face tweet_eval", default=None)
    parser.add_argument("--use-xgb", action="store_true", help="Use XGBoost (requires xgboost installed, default: True)")
    parser.add_argument("--n-iter", type=int, default=50, help="Randomized search iterations")
    parser.add_argument("--upsample", action="store_true", help="Upsample minority classes in training set (default: True)")
    parser.add_argument("--output", default="model.joblib", help="Where to save the final model (joblib)")
    # New transformer-related flags
    parser.add_argument("--use-transformer", action="store_true", help="Run transformer fine-tuning and save wrapper to model.joblib")
    parser.add_argument("--transformer-model", default="distilbert-base-uncased", help="Base model for transformer trainer")
    parser.add_argument("--transformer-output", default="transformer_model", help="Directory where transformer trainer will save model")
    parser.add_argument("--transformer-epochs", type=int, default=3)
    parser.add_argument("--transformer-batch", type=int, default=16)
    parser.add_argument("--transformer-lr", type=float, default=2e-5)
    parser.add_argument("--search-jobs", type=int, default=1, help="Parallel jobs for hyperparameter search (set >1 if system supports)")
    args = parser.parse_args()

    texts, labels, test_texts, test_labels = load_data(args.data)
    print(f"\nLoaded {len(texts)} training samples")
    counts = pd.Series(labels).value_counts().to_dict()
    print("Label distribution:", counts)

    # If user requested transformer training, call trainer then save wrapper and exit
    if args.use_transformer:
        tf_args = []
        if args.data:
            text_col = LAST_DATA_TEXT_COLUMN if LAST_DATA_TEXT_COLUMN else "text"
            label_col = LAST_DATA_LABEL_COLUMN if LAST_DATA_LABEL_COLUMN else "label"
            tf_args += ["--data", args.data, "--text-col", text_col, "--label-col", label_col]
        tf_args += ["--model", args.transformer_model,
                    "--output", args.transformer_output,
                    "--epochs", str(args.transformer_epochs),
                    "--batch", str(args.transformer_batch),
                    "--lr", str(args.transformer_lr)]
        call_transformer_trainer(tf_args)
        # Save a lightweight wrapper (pickleable) that references the transformer directory
        wrapper = TransformerWrapper(args.transformer_output)
        joblib.dump(wrapper, args.output)
        print(f"Transformer trained and wrapper saved to {args.output} (points to {args.transformer_output})")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    # Default to XGBoost and upsampling if not specified
    use_xgb = args.use_xgb if args.use_xgb else XGBOOST_AVAILABLE
    use_upsample = args.upsample if args.upsample else True
    best_model = tune_and_train(
        X_train,
        y_train,
        X_val,
        y_val,
        use_xgboost=use_xgb,
        n_iter=args.n_iter,
        upsample=use_upsample,
        n_jobs=max(1, args.search_jobs)
    )

    # Evaluate on held-out test set if available
    if test_texts is not None and test_labels is not None:
        print("\nEvaluating on held-out test set...")
        y_test_pred = best_model.predict(test_texts)
        test_acc = accuracy_score(test_labels, y_test_pred)
        test_f1 = f1_score(test_labels, y_test_pred, average='macro')
        print(f"Test Accuracy: {test_acc:.4f}  Macro F1: {test_f1:.4f}")
        print(classification_report(test_labels, y_test_pred, target_names=["Negative", "Neutral", "Positive"]))
    else:
        print("No held-out test set available.")

    joblib.dump(best_model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()

