"""
Training script for Sentiment Analysis Model
Trains a model using TF-IDF vectorization and XGBoost classifier
Saves the model as model.joblib for use in the Streamlit app
"""

import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split, cross_val_score
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
    """Generate comprehensive synthetic sentiment analysis training data with diverse examples."""
    # MASSIVELY expanded dataset with diverse contexts and vocabulary for high accuracy
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
        "Outstanding service, very happy.",
        "Brilliant! Absolutely fantastic experience.",
        "Superior quality, couldn't be happier.",
        "Exceeded all expectations, amazing!",
        "Perfect fit, exactly what I needed.",
        "Incredible value, highly recommended.",
        "Top quality product, very impressed.",
        "Outstanding customer support, thank you!",
        "Beautiful design, love everything about it.",
        "Best purchase I've made this year.",
        "Exceptional quality, worth every cent.",
        "Phenomenal product, works flawlessly.",
        "Stellar performance, very satisfied.",
        "Magnificent quality, highly pleased.",
        "Tremendous value, excellent purchase.",
        "Splendid product, very well made.",
        "Marvelous experience, highly recommend.",
        "Superb craftsmanship, excellent quality.",
        "Remarkable product, very impressed.",
        "Outstanding performance, love it!",
        "Excellent build quality, very durable.",
        "Great design, very user friendly.",
        "Perfect size, exactly as described.",
        "Fast delivery, excellent packaging.",
        "High quality materials, very sturdy.",
        "Impressive features, works great.",
        "Well designed, easy to use.",
        "Great value for money, highly satisfied.",
        "Premium quality, worth the price.",
        "Excellent product, meets all expectations.",
        "Outstanding quality, very reliable.",
        "Perfect condition, exactly as advertised.",
        "Great service, quick response.",
        "Excellent communication, very professional.",
        "Highly functional, does the job perfectly.",
        "Beautiful finish, very attractive.",
        "Smooth operation, no issues.",
        "Great performance, very efficient.",
        "Excellent quality control, perfect product.",
        "Outstanding durability, built to last.",
        "Great investment, very pleased.",
        "Excellent craftsmanship, attention to detail.",
        "Perfect functionality, works as expected.",
        "Great addition to my collection.",
        "Excellent product, highly functional.",
        "Outstanding value, great purchase.",
        "Perfect for my needs, very happy.",
        "Excellent quality, exceeds expectations.",
        "Great product, highly recommended.",
        "Outstanding performance, very satisfied.",
        # Social media style positive
        "OMG this is so good! 😍😍😍",
        "Yasss! Best thing ever! 🔥",
        "Love love love this!!! 💕",
        "So happy with this purchase! 😊",
        "This made my day! 🌟",
        "Absolutely obsessed! ❤️",
        "Can't recommend enough! ⭐⭐⭐⭐⭐",
        "This is everything! 🙌",
        "So worth it! 💯",
        "Amazing! Just amazing! ✨",
        # Review style positive
        "5 stars! Would give 10 if I could.",
        "This product changed my life for the better.",
        "I've tried many but this is the best one.",
        "Worth every single penny, highly recommend!",
        "Quality is outstanding, very impressed.",
        "Exceeded my expectations completely.",
        "Best investment I've made this year.",
        "This is exactly what I needed.",
        "Perfect solution to my problem.",
        "I'm so glad I found this product.",
        # Detailed positive
        "The quality is exceptional and the design is beautiful.",
        "Fast shipping, great packaging, excellent product.",
        "Customer service was helpful and responsive.",
        "This works exactly as advertised, very reliable.",
        "I use this daily and it never disappoints.",
        "The attention to detail is remarkable.",
        "This product has improved my daily routine.",
        "I would definitely purchase this again.",
        "The value for money is incredible.",
        "This is a game changer, love it!",
        # Emotional positive
        "This brings me so much joy!",
        "I'm thrilled with this purchase!",
        "This makes me so happy!",
        "I'm delighted with the quality!",
        "This exceeded all my hopes!",
        "I'm ecstatic about this product!",
        "This fills me with satisfaction!",
        "I'm overjoyed with this purchase!",
        "This brings me great pleasure!",
        "I'm elated with the results!",
        # Action-oriented positive
        "I will definitely buy this again!",
        "I'm telling all my friends about this!",
        "I can't wait to use this more!",
        "I'm already planning to get another one!",
        "This is going to be my go-to product!",
        "I'm recommending this to everyone!",
        "This is a must-have for everyone!",
        "I'm so excited to use this daily!",
        "This is perfect for my needs!",
        "I'm completely satisfied with this!",
        # More diverse positive expressions
        "This is incredible! Best purchase ever!",
        "I'm blown away by how good this is!",
        "This product is a masterpiece!",
        "I can't believe how amazing this is!",
        "This is pure perfection!",
        "I'm in love with this product!",
        "This exceeded all my wildest dreams!",
        "I'm speechless, it's that good!",
        "This is absolutely phenomenal!",
        "I'm so grateful I found this!",
        "This is top-tier quality!",
        "I'm amazed by the craftsmanship!",
        "This is worth its weight in gold!",
        "I'm so impressed with this!",
        "This is a dream come true!",
        "I'm beyond satisfied!",
        "This is world-class quality!",
        "I'm so glad I chose this!",
        "This is absolutely brilliant!",
        "I'm thrilled beyond words!",
        # Additional positive with strong signals
        "This is fantastic! I'm so pleased!",
        "Wonderful! Absolutely wonderful!",
        "I'm so excited about this amazing product!",
        "This is superb! Highly recommend!",
        "I'm very happy with this excellent purchase!",
        "This is great! I love it so much!",
        "I'm delighted! This is perfect!",
        "This is awesome! Best decision ever!",
        "I'm very pleased! This is wonderful!",
        "This is terrific! I'm so satisfied!"
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
        "Horrible experience, avoid this.",
        "Completely useless, total waste.",
        "Extremely poor quality, avoid at all costs.",
        "Terrible design, not functional at all.",
        "Worst experience ever, completely disappointed.",
        "Shoddy workmanship, falls apart easily.",
        "Very unreliable, breaks constantly.",
        "Poor materials, cheaply made.",
        "Defective product, doesn't work properly.",
        "Misleading description, not as advertised.",
        "Terrible customer service, unresponsive.",
        "Poor quality control, many defects.",
        "Not worth the money, overpriced junk.",
        "Very flimsy construction, breaks quickly.",
        "Terrible value, poor quality product.",
        "Completely broken, non-functional.",
        "Waste of money, regret purchasing.",
        "Poor design, difficult to use.",
        "Terrible quality, not durable at all.",
        "Very disappointed, does not meet expectations.",
        "Poor craftsmanship, sloppy work.",
        "Terrible performance, very slow.",
        "Not reliable, fails frequently.",
        "Poor build quality, feels cheap.",
        "Terrible experience, would not recommend.",
        "Very poor quality, not worth it.",
        "Completely unsatisfactory, terrible product.",
        "Poor value, low quality materials.",
        "Terrible functionality, doesn't work well.",
        "Very frustrating, constant problems.",
        "Poor design, uncomfortable to use.",
        "Terrible quality, breaks immediately.",
        "Not as described, very misleading.",
        "Poor service, unhelpful staff.",
        "Terrible product, avoid completely.",
        "Very poor construction, falls apart.",
        "Not functional, doesn't work.",
        "Terrible quality, waste of time.",
        "Poor performance, very disappointing.",
        "Completely unreliable, breaks often.",
        "Terrible value, poor quality.",
        "Not durable, breaks easily.",
        "Poor materials, low quality.",
        "Terrible experience, very unhappy.",
        "Very poor quality, not recommended.",
        "Completely defective, doesn't function.",
        "Terrible purchase, regret buying.",
        "Poor workmanship, shoddy construction.",
        "Not worth the price, overpriced.",
        "Terrible quality, avoid this product.",
        # Social media style negative
        "Ugh this is so bad 😤",
        "Worst purchase ever! 😡",
        "So disappointed! 😞",
        "This is trash! 🗑️",
        "Not worth it at all! 👎",
        "Huge waste of money! 💸",
        "This sucks! 😒",
        "Terrible experience! 😠",
        "So frustrated with this! 😤",
        "Avoid this product! ⛔",
        # Review style negative
        "1 star, would give 0 if possible.",
        "This product ruined my experience.",
        "I've never been so disappointed.",
        "Complete waste of my hard-earned money.",
        "Quality is terrible, very unimpressed.",
        "Did not meet my expectations at all.",
        "Worst investment I've made this year.",
        "This is not what I needed at all.",
        "Terrible solution, made things worse.",
        "I'm so sorry I bought this product.",
        # Detailed negative
        "The quality is poor and the design is ugly.",
        "Slow shipping, bad packaging, terrible product.",
        "Customer service was unhelpful and unresponsive.",
        "This doesn't work as advertised, very unreliable.",
        "I tried using this but it always disappoints.",
        "The lack of attention to detail is shocking.",
        "This product has made my daily routine worse.",
        "I would never purchase this again.",
        "The value for money is terrible.",
        "This is a complete waste, hate it!",
        # Emotional negative
        "This brings me so much frustration!",
        "I'm devastated with this purchase!",
        "This makes me so angry!",
        "I'm appalled with the quality!",
        "This failed all my expectations!",
        "I'm furious about this product!",
        "This fills me with disappointment!",
        "I'm heartbroken with this purchase!",
        "This brings me great sadness!",
        "I'm miserable with the results!",
        # Action-oriented negative
        "I will never buy this again!",
        "I'm warning all my friends about this!",
        "I can't wait to return this!",
        "I'm already planning to get a refund!",
        "This is going to be my worst purchase!",
        "I'm telling everyone to avoid this!",
        "This is a must-avoid for everyone!",
        "I'm so upset I have to use this!",
        "This is terrible for my needs!",
        "I'm completely unsatisfied with this!",
        # More diverse negative expressions
        "This is absolutely terrible! Worst ever!",
        "I'm shocked by how bad this is!",
        "This product is a complete disaster!",
        "I can't believe how awful this is!",
        "This is pure garbage!",
        "I'm disgusted with this product!",
        "This failed all my expectations miserably!",
        "I'm appalled, it's that bad!",
        "This is absolutely atrocious!",
        "I'm so regretful I bought this!",
        "This is bottom-tier quality!",
        "I'm horrified by the poor quality!",
        "This is worth less than nothing!",
        "I'm so unimpressed with this!",
        "This is a nightmare come true!",
        "I'm beyond disappointed!",
        "This is substandard quality!",
        "I'm so sorry I chose this!",
        "This is absolutely dreadful!",
        "I'm devastated beyond words!",
        # Additional negative with strong signals
        "This is horrible! I'm so upset!",
        "Awful! Absolutely awful!",
        "I'm so angry about this terrible product!",
        "This is dreadful! Would not recommend!",
        "I'm very unhappy with this poor purchase!",
        "This is bad! I hate it so much!",
        "I'm frustrated! This is terrible!",
        "This is lousy! Worst decision ever!",
        "I'm very displeased! This is awful!",
        "This is pathetic! I'm so unsatisfied!"
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
        "Normal product, works as expected.",
        "Standard delivery time, nothing special.",
        "Average packaging, adequate protection.",
        "Regular quality, meets basic needs.",
        "Typical product, standard features.",
        "Normal operation, no complaints.",
        "Standard design, nothing remarkable.",
        "Average performance, acceptable results.",
        "Regular item, serves its purpose.",
        "Normal quality, standard expectations.",
        "Typical experience, nothing extraordinary.",
        "Standard functionality, works adequately.",
        "Average build, acceptable quality.",
        "Regular product, meets requirements.",
        "Normal service, standard procedure.",
        "Typical quality, nothing special.",
        "Standard features, basic functionality.",
        "Average materials, acceptable construction.",
        "Regular operation, no issues.",
        "Normal condition, as expected.",
        "Standard quality, meets needs.",
        "Average performance, acceptable level.",
        "Typical product, standard quality.",
        "Normal functionality, works fine.",
        "Standard design, adequate appearance.",
        "Average experience, nothing notable.",
        "Regular quality, acceptable standard.",
        "Normal operation, no problems.",
        "Typical service, standard delivery.",
        "Standard product, meets expectations.",
        "Average quality, acceptable condition.",
        "Regular functionality, works properly.",
        "Normal performance, adequate results.",
        "Standard materials, acceptable quality.",
        "Average build, meets basic needs.",
        "Typical operation, no complaints.",
        "Normal quality, standard level.",
        "Regular service, adequate support.",
        "Standard features, basic needs met.",
        "Average condition, acceptable state.",
        "Typical quality, nothing remarkable.",
        "Normal functionality, works adequately.",
        "Standard performance, acceptable results.",
        "Average operation, no issues.",
        "Regular quality, meets standards.",
        "Normal service, standard experience.",
        "Typical product, adequate quality.",
        "Standard functionality, works as needed.",
        "Average performance, acceptable level.",
        "Regular operation, no problems.",
        "Normal quality, standard expectations.",
        # Social media style neutral
        "It's okay I guess 🤷",
        "Meh, nothing special 😑",
        "It's fine, whatever 🤔",
        "Not bad, not great 🤷‍♂️",
        "It is what it is 😐",
        "Average at best 📊",
        "Could be better, could be worse ⚖️",
        "Nothing to write home about 📝",
        "It's alright I suppose 🤷",
        "Just average, nothing more 📉",
        # Review style neutral
        "3 stars, it's acceptable but not great.",
        "This product is functional but unremarkable.",
        "I've seen better but also seen worse.",
        "It's reasonably priced for what you get.",
        "Quality is average, meets basic needs.",
        "It does the job but nothing more.",
        "Standard product with standard performance.",
        "This is adequate but not impressive.",
        "It works but I expected more.",
        "This is fine but not exceptional.",
        # Detailed neutral
        "The quality is average and the design is standard.",
        "Normal shipping, standard packaging, average product.",
        "Customer service was adequate and professional.",
        "This works as expected, reasonably reliable.",
        "I use this occasionally and it's fine.",
        "The attention to detail is standard.",
        "This product hasn't changed my routine much.",
        "I might purchase this again if needed.",
        "The value for money is reasonable.",
        "This is acceptable, no complaints.",
        # Factual neutral
        "The product arrived as scheduled.",
        "It functions according to specifications.",
        "The item matches the description provided.",
        "Standard features work as intended.",
        "The product serves its basic purpose.",
        "It meets the minimum requirements.",
        "The quality is within acceptable range.",
        "It performs at an average level.",
        "The product is functional and usable.",
        "It does what it's supposed to do.",
        # Question/uncertain neutral
        "I'm not sure how I feel about this.",
        "It might be good, need more time to decide.",
        "This could work but I'm uncertain.",
        "I'll need to use it more to judge.",
        "It seems okay but time will tell.",
        "I'm on the fence about this product.",
        "This might be useful, not sure yet.",
        "I'll give it a chance and see.",
        "It's hard to form an opinion yet.",
        "I need more experience with this to decide.",
        # More diverse neutral expressions
        "It's neither good nor bad, just average.",
        "This is a standard product with standard features.",
        "It meets the requirements but doesn't excel.",
        "This is functional but not remarkable.",
        "It's acceptable quality for the price point.",
        "This works adequately for basic needs.",
        "It's a typical product in this category.",
        "This is reasonable but not outstanding.",
        "It serves its purpose without being special.",
        "This is decent but nothing to rave about.",
        "It's a middle-of-the-road product.",
        "This is satisfactory but not impressive.",
        "It's okay for what it is.",
        "This is standard fare, nothing exceptional.",
        "It's adequate for everyday use.",
        "This is run-of-the-mill quality.",
        "It's acceptable but not noteworthy.",
        "This is typical, nothing stands out.",
        "It's fine, but I've seen better.",
        "This is ordinary, nothing special."
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
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            random_state=42,
            eval_metric='mlogloss',
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            gamma=0.05,
            reg_alpha=0.05,
            reg_lambda=0.5
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
            max_features=15000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.92,
            analyzer='word',
            lowercase=True
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
    
    # Split data with stratification for balanced classes
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
    
    # Cross-validation score for better accuracy estimate
    print("\n" + "=" * 60)
    print("Cross-Validation Score (5-fold)")
    print("=" * 60)
    try:
        cv_scores = cross_val_score(model, X_train + X_val, y_train + y_val, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Individual fold scores: {cv_scores}")
    except Exception as e:
        print(f"Cross-validation not available: {e}")
    
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

