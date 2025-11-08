"""
Test suite for Sentiment Analysis Model
Tests the trained model with various test cases
"""

import joblib
import numpy as np
import os
import sys
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
        "The item arrived on schedule."
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
            print(f"  Probabilities: Negative={proba[0]:.3f}, Neutral={proba[1]:.3f}, Positive={proba[2]:.3f}")
            print(f"  Sum: {prob_sum:.3f} {'VALID' if is_valid else 'INVALID'}")
            
            if not is_valid:
                passed = False
        else:
            print(f"WARNING: Model does not support predict_proba")
            passed = False
            break
    
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_batch_predictions(model):
    """Test batch predictions."""
    print("\n" + "=" * 60)
    print("Test 3: Batch Predictions")
    print("=" * 60)
    
    test_texts = [
        "Great product!",
        "Not bad, not great.",
        "Awful quality.",
        "I'm very happy with this purchase.",
        "It's okay, nothing special.",
        "Terrible experience.",
        "Excellent service!",
        "Average product.",
        "Love it!",
        "Hate it!"
    ]
    
    predictions = model.predict(test_texts)
    
    print(f"Processed {len(test_texts)} texts")
    print(f"Predictions: {[LABEL_MAP[p] for p in predictions]}")
    
    # Check that all predictions are valid (0, 1, or 2)
    valid_predictions = all(p in [0, 1, 2] for p in predictions)
    
    print(f"\nResult: {'PASS' if valid_predictions else 'FAIL'}")
    return valid_predictions


def test_edge_cases(model):
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Test 4: Edge Cases")
    print("=" * 60)
    
    edge_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        "a",  # Single character
        "This is a very long text. " * 100,  # Very long text
        "123456789",  # Numbers only
        "!@#$%^&*()",  # Special characters only
        "I'm happy!",  # With punctuation (emoji removed for Windows compatibility)
        "UPPERCASE TEXT",  # All uppercase
        "lowercase text",  # All lowercase
        "MiXeD cAsE tExT",  # Mixed case
    ]
    
    passed = True
    
    for text in edge_cases:
        try:
            prediction = model.predict([text])[0]
            predicted_label = LABEL_MAP[prediction]
            
            # Check that prediction is valid
            if prediction not in [0, 1, 2]:
                # Safe text display for Windows console
                safe_text = text[:30].encode('ascii', 'ignore').decode('ascii') if text else ""
                print(f"FAIL | Text: '{safe_text}...' | Invalid prediction: {prediction}")
                passed = False
            else:
                # Safe text display for Windows console
                safe_text = text[:30].encode('ascii', 'ignore').decode('ascii') if text else ""
                print(f"PASS | Text: '{safe_text}...' | Prediction: {predicted_label}")
        except Exception as e:
            # Safe text display for Windows console
            safe_text = text[:30].encode('ascii', 'ignore').decode('ascii') if text else ""
            error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            print(f"FAIL | Text: '{safe_text}...' | Error: {error_msg}")
            passed = False
    
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_model_consistency(model):
    """Test model consistency (same input should give same output)."""
    print("\n" + "=" * 60)
    print("Test 5: Model Consistency")
    print("=" * 60)
    
    test_text = "I love this product!"
    
    predictions = []
    for _ in range(5):
        pred = model.predict([test_text])[0]
        predictions.append(pred)
    
    # All predictions should be the same
    is_consistent = len(set(predictions)) == 1
    
    print(f"Text: '{test_text}'")
    print(f"Predictions over 5 runs: {[LABEL_MAP[p] for p in predictions]}")
    print(f"\nResult: {'PASS' if is_consistent else 'FAIL'}")
    
    return is_consistent


def test_integration_with_app_format(model):
    """Test that model works with app.py format."""
    print("\n" + "=" * 60)
    print("Test 6: Integration with App Format")
    print("=" * 60)
    
    # Simulate how app.py uses the model
    test_texts = ["I love this!", "This is terrible.", "It's okay."]
    
    try:
        # Test predict method
        predictions = model.predict(test_texts)
        assert len(predictions) == len(test_texts), "Prediction length mismatch"
        
        # Test predict_proba if available
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(test_texts)
            assert probas.shape[0] == len(test_texts), "Proba shape mismatch"
            assert probas.shape[1] == 3, "Should have 3 classes"
        
        print("PASS: Model interface matches app.py requirements")
        print(f"   - predict() works: YES")
        print(f"   - predict_proba() works: {'YES' if hasattr(model, 'predict_proba') else 'NO (optional)'}")
        
        return True
    except Exception as e:
        print(f"FAIL: Model interface mismatch: {e}")
        return False


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("SENTIMENT ANALYSIS MODEL TEST SUITE")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists("model.joblib"):
        print("\nERROR: Model file 'model.joblib' not found!")
        print("Please run 'python train_model.py' first to train the model.")
        return False
    
    # Load model
    try:
        print("\nLoading model...")
        model = load_model()
        print("SUCCESS: Model loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False
    
    # Run tests
    test_results = []
    
    test_results.append(("Basic Predictions", test_basic_predictions(model)))
    test_results.append(("Prediction Probabilities", test_prediction_proba(model)))
    test_results.append(("Batch Predictions", test_batch_predictions(model)))
    test_results.append(("Edge Cases", test_edge_cases(model)))
    test_results.append(("Model Consistency", test_model_consistency(model)))
    test_results.append(("App Integration", test_integration_with_app_format(model)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{status} | {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\nSUCCESS: ALL TESTS PASSED! Model is ready to use.")
        return True
    else:
        print(f"\nWARNING: {total - passed} test(s) failed. Please review the results.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

