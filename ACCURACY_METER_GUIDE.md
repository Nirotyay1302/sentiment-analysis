# 📊 Accuracy Meter & Validation Guide

## Overview

The Accuracy Meter feature allows you to validate your sentiment analysis model by comparing predicted sentiments with actual labels from a reference dataset.

## Features

### ✅ Comprehensive Metrics
- **Overall Accuracy**: Percentage of correct predictions
- **Per-Class Metrics**: Precision, Recall, F1-Score for each sentiment class
- **Macro Averages**: Overall precision, recall, and F1-score
- **Confusion Matrix**: Visual representation of prediction accuracy
- **Classification Report**: Detailed text report

### ✅ Detailed Comparison
- Side-by-side comparison of actual vs predicted labels
- Filter by correct/incorrect predictions
- Download results as CSV

## How to Use

### Step 1: Prepare Your Reference Dataset

Your CSV file should have:
- **Text column**: Contains the text to analyze
- **Sentiment/Label column**: Contains actual labels (Positive, Negative, Neutral)

**Example CSV format**:
```csv
text,sentiment
"I love this product!",Positive
"This is terrible.",Negative
"The product arrived on time.",Neutral
```

### Step 2: Upload Reference Dataset

1. Go to **"Accuracy Meter/Validation"** mode
2. Click **"Upload reference dataset"**
3. Select your CSV file
4. The app will auto-detect text and sentiment columns

### Step 3: Select Columns

1. **Text column**: Select the column containing text data
2. **Sentiment column**: Select the column containing actual labels

The app will show a preview of your data.

### Step 4: Calculate Accuracy

1. Click **"Calculate Accuracy"** button
2. The app will:
   - Predict sentiments for all texts
   - Compare with actual labels
   - Calculate comprehensive metrics
   - Display results

### Step 5: Review Results

View:
- **Overall Accuracy**: Main accuracy percentage
- **Per-Class Metrics**: Precision, Recall, F1-Score for each class
- **Confusion Matrix**: Visual matrix showing correct/incorrect predictions
- **Detailed Comparison**: Table showing actual vs predicted for each text
- **Classification Report**: Detailed metrics report

### Step 6: Download Results

Download a CSV file with:
- Original text
- Actual label
- Predicted label
- Match status (True/False)

## Understanding Metrics

### Overall Accuracy
- **Formula**: (Correct Predictions / Total Samples) × 100
- **Range**: 0% to 100%
- **Interpretation**: Higher is better

### Precision
- **Definition**: Of all predictions for a class, how many were correct
- **Formula**: True Positives / (True Positives + False Positives)
- **Interpretation**: Higher precision = fewer false positives

### Recall
- **Definition**: Of all actual instances of a class, how many were found
- **Formula**: True Positives / (True Positives + False Negatives)
- **Interpretation**: Higher recall = fewer false negatives

### F1-Score
- **Definition**: Harmonic mean of precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Interpretation**: Balanced metric considering both precision and recall

### Confusion Matrix
Shows:
- **Diagonal**: Correct predictions (True Positives)
- **Off-diagonal**: Incorrect predictions (False Positives/Negatives)

## Use Cases

### 1. Model Validation
- Test your trained model on a validation dataset
- Measure model performance
- Identify areas for improvement

### 2. Model Comparison
- Compare different models
- Test model updates
- Evaluate model improvements

### 3. Quality Assurance
- Validate model accuracy before deployment
- Check model performance on specific datasets
- Ensure model meets accuracy requirements

### 4. Training Data Evaluation
- Test if training data is representative
- Identify data quality issues
- Guide data collection efforts

## Best Practices

### Dataset Preparation
1. **Balanced Dataset**: Include all three sentiment classes
2. **Representative Data**: Use data similar to production use cases
3. **Quality Labels**: Ensure labels are accurate and consistent
4. **Adequate Size**: Use at least 50-100 samples for reliable metrics

### Label Format
- Use: "Positive", "Negative", "Neutral" (case-insensitive)
- Avoid: Mixed formats, typos, inconsistent labels
- Normalize: The app automatically normalizes labels

### Interpreting Results
- **Accuracy > 80%**: Excellent model performance
- **Accuracy 60-80%**: Good performance, may need improvement
- **Accuracy < 60%**: Consider retraining or improving the model

### Improving Accuracy
1. **More Training Data**: Add more labeled examples
2. **Better Features**: Improve text preprocessing
3. **Model Tuning**: Adjust hyperparameters
4. **Data Quality**: Ensure training data is high quality

## Troubleshooting

### Invalid Labels Warning
**Problem**: App shows warning about invalid labels

**Solution**:
- Ensure labels are: Positive, Negative, or Neutral
- Check for typos or extra spaces
- Normalize labels before uploading

### Low Accuracy
**Problem**: Model shows low accuracy

**Solutions**:
1. Check if reference dataset matches training data distribution
2. Verify labels are correct
3. Consider retraining with more data
4. Check for class imbalance

### Missing Metrics
**Problem**: Some metrics not showing

**Solution**:
- Ensure scikit-learn is installed
- Check that dataset has all three classes
- Verify sufficient samples per class

## Example Workflow

1. **Train Model**: Use `train_model.py` to train your model
2. **Prepare Test Data**: Create CSV with text and actual labels
3. **Upload & Validate**: Use Accuracy Meter to test model
4. **Review Metrics**: Check accuracy, precision, recall
5. **Improve Model**: Retrain if needed based on results
6. **Re-validate**: Test again with updated model

## CSV Format Examples

### Simple Format
```csv
text,sentiment
"Great product!",Positive
"Not good.",Negative
"Okay.",Neutral
```

### With Multiple Columns
```csv
id,review_text,actual_sentiment,date
1,"I love it!",Positive,2024-01-01
2,"Terrible quality",Negative,2024-01-02
3,"It's fine",Neutral,2024-01-03
```

The app will auto-detect the text and sentiment columns.

## Tips

1. **Start Small**: Test with 50-100 samples first
2. **Check Confusion Matrix**: Identify which classes are confused
3. **Review Incorrect Predictions**: Learn from mistakes
4. **Compare Models**: Test different models on same dataset
5. **Track Over Time**: Monitor accuracy as you improve the model

## Limitations

- Requires labeled reference dataset
- Accuracy depends on reference data quality
- Metrics are only as good as your labels
- Small datasets may show high variance

## Next Steps

After validation:
1. **If accuracy is good**: Deploy the model
2. **If accuracy is low**: Retrain with more/better data
3. **If specific classes fail**: Focus on improving those classes
4. **If confusion matrix shows patterns**: Address systematic errors

---

**Use the Accuracy Meter to ensure your model is ready for production! 📊✅**

