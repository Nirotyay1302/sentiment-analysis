import argparse
import os
import pandas as pd
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Try to import XGBoost, fall back to LogisticRegression if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using LogisticRegression")

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def textblob_label(text: str) -> str:
    tb = TextBlob(text)
    p = tb.sentiment.polarity
    if p > 0.05:
        return "positive"
    if p < -0.05:
        return "negative"
    return "neutral"


def load_dataset(path: str):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # try to locate text column
    text_col = None
    for c in df.columns:
        if "text" in c.lower():
            text_col = c
            break
    if text_col is None:
        # fallback to first column
        text_col = df.columns[0]
    df = df[[text_col] + [c for c in df.columns if c != text_col]]
    df = df.rename(columns={text_col: "text"})
    return df


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "label" in df.columns:
        # normalize label strings and map
        df["label_str"] = df["label"].astype(str).str.strip().str.lower()
        # if labels are numeric strings 0/1/2, try to keep them
        if df["label_str"].dropna().apply(lambda x: x.isdigit()).all():
            df["label_num"] = df["label_str"].astype(int)
        else:
            df["label_num"] = df["label_str"].map(LABEL_MAP)
    else:
        # pseudo-label using TextBlob
        df["label_str"] = df["text"].astype(str).apply(textblob_label)
        df["label_num"] = df["label_str"].map(LABEL_MAP)
    # drop rows where mapping failed
    df = df[df["label_num"].notnull()].copy()
    df["label_num"] = df["label_num"].astype(int)
    return df


def train_and_save(df, output_path: str, test_size=0.2, random_state=42):
    df["text_clean"] = df["text"].astype(str).apply(clean_text)
    X = df["text_clean"].tolist()
    y = df["label_num"].tolist()

    # Only stratify if we have enough samples to guarantee each class gets at least 2 samples in train/test
    # Since test_size=0.2, we need at least 10 samples per class to be safe
    # For small datasets, don't stratify to avoid splitting errors
    unique_classes = len(set(y))
    should_stratify = len(y) >= 30 and unique_classes > 1  # 30 samples minimum for safe stratification
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if should_stratify else None
    )

    # Use XGBoost if available, otherwise LogisticRegression
    if XGBOOST_AVAILABLE:
        classifier = XGBClassifier(
            random_state=random_state,
            eval_metric='mlogloss',
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100
        )
        print("Training with XGBoost classifier")
    else:
        classifier = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
        print("Training with LogisticRegression classifier")
    
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")),
            ("clf", classifier),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"], zero_division=0))

    # Save pipeline to output_path
    joblib.dump(pipe, output_path)
    print(f"Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train sentiment model and save model.joblib")
    parser.add_argument("--input", "-i", type=str, default="data/labeled_data.csv", help="Path to input CSV (text,label). If missing, tries data/unlabeled.csv or falls back to sample data.")
    parser.add_argument("--output", "-o", type=str, default="model.joblib", help="Output model path (model.joblib)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    args = parser.parse_args()

    input_path = args.input
    df = None
    if os.path.exists(input_path):
        print(f"Loading dataset from {input_path}")
        df = load_dataset(input_path)
    else:
        alt = "data/unlabeled.csv"
        if os.path.exists(alt):
            print(f"{input_path} not found; loading unlabeled file {alt} and pseudo-labeling with TextBlob")
            df = load_dataset(alt)
        else:
            print("No input CSV found. Training on a small built-in example dataset.")
            print("WARNING: This is a demo dataset. For production use, provide a real dataset with hundreds or thousands of examples.")
            sample = [
                # Positive examples
                ("I love this product, it's amazing!", "positive"),
                ("Absolutely fantastic experience!", "positive"),
                ("This is wonderful and delightful!", "positive"),
                ("Perfect! Couldn't ask for more.", "positive"),
                ("Excellent quality, highly recommend!", "positive"),
                ("Outstanding service and support!", "positive"),
                ("Brilliant! Best purchase ever!", "positive"),
                ("Super happy with the results!", "positive"),
                ("Great product, very satisfied!", "positive"),
                ("Love it! Works perfectly for me.", "positive"),
                ("Amazing value for money!", "positive"),
                ("This makes me so happy!", "positive"),
                # Negative examples
                ("This is the worst, totally disappointed.", "negative"),
                ("Terrible service, will not return.", "negative"),
                ("Hate this product, waste of money.", "negative"),
                ("Poor quality, very unhappy.", "negative"),
                ("Awful experience, don't recommend.", "negative"),
                ("Horrible service, never again!", "negative"),
                ("Disappointed beyond words.", "negative"),
                ("This is garbage, returns!", "negative"),
                ("Worst purchase of my life.", "negative"),
                ("Regret buying this completely.", "negative"),
                ("Not worth the price at all.", "negative"),
                ("Very bad experience, stay away!", "negative"),
                # Neutral examples
                ("Not bad, could be better.", "neutral"),
                ("It was okay, nothing special.", "neutral"),
                ("Average product, does the job.", "neutral"),
                ("Fine I guess, nothing to complain.", "neutral"),
                ("It's alright, meets expectations.", "neutral"),
                ("Standard quality, no issues.", "neutral"),
                ("Decent product, got what expected.", "neutral"),
                ("Fair quality, acceptable for price.", "neutral"),
                ("Okay performance, middle ground.", "neutral"),
                ("Average value, not bad not great.", "neutral"),
            ]
            df = pd.DataFrame(sample, columns=["text", "label"])

    df = prepare_labels(df)
    if df.empty:
        raise SystemExit("No usable labeled rows after processing. Provide a CSV with text and label or a valid unlabeled CSV to pseudo-label.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    train_and_save(df, args.output, test_size=args.test_size)


if __name__ == "__main__":
    main()