"""
Test suite for Sentiment Analysis Model
Tests the trained model with various test cases
"""

import joblib
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Label mapping
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

def load_model(model_path="model.joblib"):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
    return joblib.load(model_path)

def test_basic_predictions(model):
    """Test basic sentiment predictions."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Sentiment Predictions")
    print("=" * 60)
    
    test_cases = [
        ("I love this product! It's amazing!", 2),  # Positive
        ("This is terrible. Worst purchase ever.", 0),  # Negative
        ("The product arrived on time.", 1),  # Neutral
        ("Excellent quality, highly recommended!", 2),  # Positive
        ("Very disappointed with the service.", 0),  # Negative
        ("It works as expected, nothing special.", 1),  # Neutral
        ("Absolutely fantastic experience!", 2),  # Positive
        ("I hate this!", 0),  # Negative
        ("It's okay, I guess.", 1),  # Neutral
        ("Best purchase I've ever made!", 2),  # Positive
        ("Worst decision ever.", 0),  # Negative
        ("Not bad, not great.", 1),  # Neutral
    ]
    
    passed = 0
    total = len(test_cases)
    
    for text, expected_label in test_cases:
        prediction = model.predict([text])[0]
        predicted_label = LABEL_MAP[prediction]
        expected_label_name = LABEL_MAP[expected_label]
        
        status = "PASS" if prediction == expected_label else "FAIL"
        print(f"{status} | Text: '{text[:50]}...'")
        print(f"      Expected: {expected_label_name}, Got: {predicted_label}")
        
        if prediction == expected_label:
            passed += 1
    
    print(f"\nResult: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    return passed == total

def test_prediction_proba(model):
    """Test prediction probabilities."""
    print("\n" + "=" * 60)
    print("Test 2: Prediction Probabilities")
    print("=" * 60)
    
    test_texts = [
        "I absolutely love this!",
        "This is terrible.",
        "The item arrived on schedule.",
        "I feel indifferent about this.",
        "This is the best thing ever!",
        "I really dislike this product."
    ]
    
    passed = True
    
    for text in test_texts:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([text])[0]
            prediction = model.predict([text])[0]
            
            # Check that probabilities sum to ~1.0
            prob_sum = np.sum(proba)
            is_valid = abs(prob_sum - 1.0) < 0.01 and len(proba) == 3
            
            print(f"Text: '{text}'")
            print(f"  Prediction: {LABEL_MAP[prediction]}")
            print(f"  Probabilities: {proba}, Valid: {is_valid}")

            if not is_valid:
                passed = False
        else:
            print(f"Model does not support probability predictions for text: '{text}'")
            passed = False

    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed

def test_batch_predictions(model):
    """Test batch predictions."""
    print("\n" + "=" * 60)
    print("Test 3: Batch Predictions")
    print("=" * 60)
    
    test_texts = [
        "I love this product!",
        "This is the worst thing ever.",
        "It's okay, not great but not bad.",
        "Fantastic service, will buy again!",
        "I regret this purchase.",
        "Mediocre at best."
    ]
    
    predictions = model.predict(test_texts)
    for text, prediction in zip(test_texts, predictions):
        print(f"Text: '{text}' | Prediction: {LABEL_MAP[prediction]}")

def test_model_performance_on_sample(model):
    """Evaluate model performance on a sample dataset and print detailed metrics."""
    print("\n" + "=" * 60)
    print("Test 4: Detailed Performance on Sample Data")
    print("=" * 60)

    test_cases = [
        ("I love this product! It's amazing!", 2),
        ("This is terrible. Worst purchase ever.", 0),
        ("The product arrived on time.", 1),
        ("Excellent quality, highly recommended!", 2),
        ("Very disappointed with the service.", 0),
        ("It works as expected, nothing special.", 1),
        ("Absolutely fantastic experience!", 2),
        ("I hate this!", 0),
        ("It's okay, I guess.", 1),
        ("Best purchase I've ever made!", 2),
        ("Worst decision ever.", 0),
        ("Not bad, not great.", 1),
    ]

    texts = [tc[0] for tc in test_cases]
    true_labels = [tc[1] for tc in test_cases]

    # Get model predictions
    predicted_labels = model.predict(texts)

    # Calculate and print metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy on sample data: {accuracy:.2%}\n")

    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=LABEL_MAP.values()))

    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

def main():
    model = load_model("model.joblib")
    test_basic_predictions(model)
    test_prediction_proba(model)
    test_batch_predictions(model)
    test_model_performance_on_sample(model)

if __name__ == "__main__":
    main()