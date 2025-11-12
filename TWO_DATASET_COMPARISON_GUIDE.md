# 🔄 Two Dataset Comparison Guide

## Overview

The "Compare Two Datasets" feature allows you to compare a **labeled dataset** (with declared sentiments) against an **unlabeled dataset** (non-declared) to measure model accuracy and performance.

## Features

### ✅ Dataset Comparison
- Compare labeled vs unlabeled datasets
- Two matching methods: Row Index or Text Similarity
- Automatic numerical column exclusion
- Comprehensive accuracy metrics

### ✅ Metrics Provided
- Overall accuracy percentage
- Per-class metrics (Precision, Recall, F1-Score)
- Confusion matrix visualization
- Detailed comparison table
- Downloadable results

## How to Use

### Step 1: Prepare Your Datasets

#### Reference Dataset (With Labels)
Your CSV should have:
- **Text column**: Contains text data
- **Sentiment/Label column**: Contains actual labels (Positive, Negative, Neutral)

**Example**:
```csv
text,sentiment
"I love this product!",Positive
"This is terrible.",Negative
"The product arrived on time.",Neutral
```

#### Test Dataset (Without Labels)
Your CSV should have:
- **Text column**: Contains text data (no sentiment column needed)

**Example**:
```csv
text
"I really like this!"
"This is awful."
"It's okay."
```

### Step 2: Upload Datasets

1. Go to **"Compare Two Datasets"** mode
2. **Upload Reference Dataset**: CSV with text and sentiment labels
3. **Upload Test Dataset**: CSV with text only (no labels)

### Step 3: Select Columns

1. **Reference Dataset**:
   - Select text column (numerical columns automatically excluded)
   - Select sentiment/label column

2. **Test Dataset**:
   - Select text column (numerical columns automatically excluded)

The app will show previews of both datasets.

### Step 4: Choose Matching Method

Select how to match the datasets:

#### Option 1: By Row Index (Same Order)
- Matches rows by position (row 1 with row 1, etc.)
- Use when datasets are in the same order
- Faster processing
- Best for: Datasets with same structure and order

#### Option 2: By Text Similarity
- Matches texts by similarity (70% threshold)
- Use when datasets have different orders
- Slower but more flexible
- Best for: Datasets with similar but not identical texts

### Step 5: Compare & View Results

1. Click **"Compare Datasets & Calculate Accuracy"**
2. View results:
   - Overall accuracy
   - Per-class metrics
   - Confusion matrix
   - Detailed comparison table
3. Download results as CSV

## Understanding the Results

### Overall Accuracy
- Percentage of correct predictions
- Formula: (Correct / Total) × 100

### Per-Class Metrics
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Balanced metric

### Confusion Matrix
Shows:
- Correct predictions (diagonal)
- Incorrect predictions (off-diagonal)
- Which classes are confused

### Detailed Comparison
- Side-by-side view of:
  - Test text
  - Reference label (actual)
  - Predicted label
  - Match status (✅ or ❌)

## Numerical Column Exclusion

### Automatic Detection
The app automatically:
- ✅ Detects numerical columns
- ✅ Excludes them from text analysis
- ✅ Shows which columns were excluded
- ✅ Only shows text columns for selection

### Why This Matters
- Prevents analyzing IDs, timestamps, scores, etc.
- Ensures only text data is analyzed
- Improves accuracy by focusing on relevant data

### Example
If your CSV has:
```csv
id,text,score,sentiment
1,"I love it!",4.5,Positive
2,"Terrible",1.2,Negative
```

The app will:
- ✅ Exclude: `id`, `score` (numerical)
- ✅ Show: `text`, `sentiment` (text columns)

## Use Cases

### 1. Model Validation
- Test model on new unlabeled data
- Compare with known labels
- Measure real-world performance

### 2. Dataset Quality Check
- Compare two versions of a dataset
- Check if labels are consistent
- Identify labeling errors

### 3. Model Comparison
- Test different models on same data
- Compare accuracy across models
- Choose best performing model

### 4. Production Monitoring
- Compare predictions with ground truth
- Monitor model drift
- Track accuracy over time

## Matching Methods Explained

### By Row Index
**When to use**: Datasets are in the same order

**How it works**:
- Row 1 of reference → Row 1 of test
- Row 2 of reference → Row 2 of test
- And so on...

**Limitations**:
- Requires same order
- Different lengths: compares first N rows

### By Text Similarity
**When to use**: Datasets have different orders or slightly different texts

**How it works**:
- For each test text, finds most similar reference text
- Uses 70% similarity threshold
- Matches best similar texts

**Limitations**:
- Slower processing
- May not find matches if texts are too different
- Requires at least 70% similarity

## Best Practices

### Dataset Preparation
1. **Clean Data**: Remove duplicates, handle missing values
2. **Consistent Format**: Use same text format in both datasets
3. **Label Quality**: Ensure reference labels are accurate
4. **Adequate Size**: Use at least 50-100 samples for reliable metrics

### Matching Method Selection
- **Same Order**: Use "By Row Index" (faster)
- **Different Order**: Use "By Text Similarity" (more flexible)
- **Different Texts**: Ensure texts are similar enough (70% threshold)

### Interpreting Results
- **Accuracy > 80%**: Excellent model performance
- **Accuracy 60-80%**: Good, may need improvement
- **Accuracy < 60%**: Consider retraining or improving model

## Troubleshooting

### No Matches Found (Text Similarity)
**Problem**: No texts matched with 70% similarity

**Solutions**:
1. Try "By Row Index" method instead
2. Check if texts are similar enough
3. Ensure datasets contain related texts
4. Lower similarity threshold (if possible)

### Different Dataset Lengths
**Problem**: Datasets have different number of rows

**Solution**:
- App automatically compares first N rows (where N = minimum length)
- Warning message shows how many rows compared

### Low Accuracy
**Problem**: Model shows low accuracy

**Solutions**:
1. Check if reference labels are correct
2. Verify datasets are properly matched
3. Ensure model is trained on similar data
4. Consider retraining with more data

### Numerical Columns Detected
**Problem**: App excludes columns you want to use

**Solution**:
- Numerical columns are automatically excluded
- This is intentional to prevent analyzing numbers
- Use text columns only for sentiment analysis

## CSV Format Examples

### Reference Dataset (With Labels)
```csv
text,sentiment
"Great product!",Positive
"Not good",Negative
"Okay",Neutral
```

### Test Dataset (Without Labels)
```csv
text
"Amazing quality!"
"Poor service"
"Average"
```

### With Multiple Columns
```csv
id,review_text,rating,actual_sentiment
1,"I love it!",5,Positive
2,"Terrible",1,Negative
3,"It's fine",3,Neutral
```

The app will:
- Exclude: `id`, `rating` (numerical)
- Use: `review_text`, `actual_sentiment` (text)

## Tips

1. **Start Small**: Test with 50-100 samples first
2. **Check Matching**: Verify datasets are properly matched
3. **Review Incorrect**: Learn from prediction errors
4. **Use Appropriate Method**: Choose matching method based on your data
5. **Validate Labels**: Ensure reference labels are accurate

## Limitations

- Requires labeled reference dataset
- Matching may not be perfect (especially with text similarity)
- Accuracy depends on reference label quality
- Text similarity matching has 70% threshold

## Next Steps

After comparison:
1. **If accuracy is good**: Model is performing well
2. **If accuracy is low**: Review incorrect predictions, improve model
3. **If specific classes fail**: Focus on improving those classes
4. **If matching fails**: Try different matching method or check data

---

**Use Two Dataset Comparison to validate your model on new data! 🔄📊**

