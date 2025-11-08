"""
Training script for Sentiment Analysis Model
Trains a model using TF-IDF vectorization and XGBoost classifier
Saves the model as model.joblib for use in the Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import os

# Label mapping: 0=Negative, 1=Neutral, 2=Positive
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
REVERSE_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


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
    Load training data from CSV file or generate synthetic data for demonstration.
    
    Expected CSV format:
    - Column with text data (auto-detected if column name contains 'text')
    - Column with sentiment labels: 'Negative', 'Neutral', or 'Positive'
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Auto-detect text and label columns
        text_col = None
        label_col = None
        
        for col in df.columns:
            if 'text' in col.lower() or 'comment' in col.lower() or 'review' in col.lower():
                text_col = col
            if 'sentiment' in col.lower() or 'label' in col.lower():
                label_col = col
        
        if text_col is None:
            # Use the first string column as text
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
        
        if label_col is None:
            raise ValueError("Could not find label column. Expected column with 'sentiment' or 'label' in name.")
        
        texts = df[text_col].apply(clean_text).tolist()
        labels = df[label_col].apply(lambda x: LABEL_MAP.get(str(x).strip(), 1)).tolist()
        
        return texts, labels
    else:
        # Generate synthetic training data for demonstration
        print("No data file provided. Generating synthetic training data...")
        return generate_synthetic_data()


def generate_synthetic_data():
    """Generate synthetic sentiment analysis training data."""
    # Expanded dataset for better training
    positive_texts = [
        "I love this product! It's amazing and works perfectly.",
        "Great service, very satisfied with my purchase.",
        "Excellent quality, highly recommend to everyone.",
        "This is the best thing I've ever bought!",
        "Outstanding performance, exceeded my expectations.",
        "Wonderful experience, will definitely buy again.",
        "Fantastic product, worth every penny.",
        "Amazing quality, very happy with this purchase.",
        "Perfect! Exactly what I was looking for.",
        "Highly satisfied, great value for money.",
        "Love it! Great customer service too.",
        "Excellent product, fast shipping.",
        "Really happy with this purchase, great quality.",
        "Outstanding! Better than I expected.",
        "Wonderful product, highly recommended.",
        "Fantastic! Works perfectly as described.",
        "Amazing experience, very pleased.",
        "Perfect solution for my needs.",
        "Great product, excellent quality.",
        "Love this! Will definitely order again.",
        "Superb quality, excellent craftsmanship.",
        "Brilliant product, highly impressed.",
        "Top notch quality, very satisfied.",
        "Exceptional service, great experience.",
        "Outstanding value, would buy again.",
        "Excellent quality, very pleased.",
        "Great purchase, highly recommend.",
        "Perfect product, exceeds expectations.",
        "Amazing quality, love it!",
        "Fantastic service, very happy.",
        "Wonderful product, great value.",
        "Excellent purchase, very satisfied.",
        "Outstanding quality, highly recommend.",
        "Great product, works perfectly.",
        "Perfect quality, very pleased.",
        "Amazing product, excellent value.",
        "Fantastic quality, love it!",
        "Wonderful purchase, highly satisfied.",
        "Excellent product, great quality.",
        "Outstanding service, very happy."
    ]
    
    negative_texts = [
        "Terrible product, waste of money.",
        "Very disappointed with this purchase.",
        "Poor quality, broke after one use.",
        "Awful experience, would not recommend.",
        "Horrible service, worst purchase ever.",
        "Bad quality, not worth the price.",
        "Disappointing product, does not work.",
        "Terrible customer service, very unhappy.",
        "Poor quality control, defective item.",
        "Worst product I've ever bought.",
        "Very poor quality, complete waste.",
        "Disappointed, did not meet expectations.",
        "Bad experience, would not buy again.",
        "Terrible quality, broken on arrival.",
        "Awful product, money down the drain.",
        "Horrible, worst purchase decision.",
        "Poor service, very dissatisfied.",
        "Bad product, does not function properly.",
        "Terrible, regret buying this.",
        "Very poor, not recommended at all.",
        "Completely broken, useless product.",
        "Waste of money, poor quality.",
        "Very bad experience, disappointed.",
        "Terrible product, does not work.",
        "Poor quality, regret purchasing.",
        "Awful service, very unhappy.",
        "Bad product, not as described.",
        "Terrible quality, broken immediately.",
        "Very poor, waste of time.",
        "Horrible product, avoid at all costs.",
        "Disappointing quality, poor value.",
        "Bad experience, would not recommend.",
        "Terrible service, very dissatisfied.",
        "Poor product, does not function.",
        "Awful quality, complete disappointment.",
        "Very bad, worst purchase ever.",
        "Terrible product, money wasted.",
        "Poor quality, broken quickly.",
        "Bad service, very disappointed.",
        "Horrible experience, avoid this."
    ]
    
    neutral_texts = [
        "The product arrived on time.",
        "It is what it is, nothing special.",
        "Average quality, works as expected.",
        "Standard product, meets basic requirements.",
        "Received the item, looks okay.",
        "Product is fine, nothing extraordinary.",
        "Decent quality, average performance.",
        "It works, but nothing impressive.",
        "Standard item, does the job.",
        "Average product, acceptable quality.",
        "Received order, seems fine.",
        "Normal product, nothing to complain about.",
        "Basic quality, serves its purpose.",
        "Average experience, okay for the price.",
        "Standard quality, meets expectations.",
        "Product works, nothing special.",
        "Decent item, average performance.",
        "Normal quality, acceptable.",
        "Standard product, does what it should.",
        "Average item, no complaints.",
        "Product delivered as expected.",
        "Standard quality, meets requirements.",
        "Average product, works fine.",
        "Normal item, acceptable quality.",
        "Basic product, does the job.",
        "Standard quality, nothing special.",
        "Average performance, acceptable.",
        "Normal product, meets expectations.",
        "Basic item, works as described.",
        "Standard quality, fine for the price.",
        "Average product, acceptable quality.",
        "Normal item, works okay.",
        "Basic quality, serves purpose.",
        "Standard product, meets needs.",
        "Average quality, acceptable.",
        "Normal item, does the job.",
        "Basic product, works fine.",
        "Standard quality, acceptable.",
        "Average item, meets requirements.",
        "Normal product, works as expected."
    ]
    
    # Combine and create labels
    texts = positive_texts + negative_texts + neutral_texts
    labels = [2] * len(positive_texts) + [0] * len(negative_texts) + [1] * len(neutral_texts)
    
    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return texts, labels


def train_model(X_train, y_train, X_val, y_val, use_xgboost=True):
    """
    Train sentiment analysis model.
    
    Args:
        X_train: Training texts
        y_train: Training labels (0=Negative, 1=Neutral, 2=Positive)
        X_val: Validation texts
        y_val: Validation labels
        use_xgboost: If True, use XGBoost; else use Logistic Regression
    
    Returns:
        Trained pipeline model
    """
    print(f"\nTraining {'XGBoost' if use_xgboost else 'Logistic Regression'} model...")
    
    # Create pipeline with TF-IDF vectorization and classifier
    if use_xgboost:
        classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
    else:
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )),
        ('classifier', classifier)
    ])
    
    # Train the model
    print("Fitting model on training data...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Negative", "Neutral", "Positive"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    return pipeline


def main():
    """Main training function."""
    print("=" * 60)
    print("Sentiment Analysis Model Training")
    print("=" * 60)
    
    # Load data
    # You can specify a data file: load_data("data/training_data.csv")
    texts, labels = load_data()
    
    print(f"\nLoaded {len(texts)} training samples")
    print(f"Label distribution:")
    for label, count in zip(["Negative", "Neutral", "Positive"], 
                            [labels.count(0), labels.count(1), labels.count(2)]):
        print(f"  {label}: {count}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, use_xgboost=True)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Final Test Set Evaluation")
    print("=" * 60)
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Negative", "Neutral", "Positive"]))
    
    # Save model
    model_path = "model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSUCCESS: Model saved to {model_path}")
    
    # Test model loading
    print("\nTesting model loading...")
    loaded_model = joblib.load(model_path)
    test_pred = loaded_model.predict(["I love this product!"])
    print(f"Test prediction: {REVERSE_LABEL_MAP[test_pred[0]]}")
    print("SUCCESS: Model loading test passed!")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nModel file: {model_path}")
    print("You can now use this model in the Streamlit app (app.py)")


if __name__ == "__main__":
    main()

