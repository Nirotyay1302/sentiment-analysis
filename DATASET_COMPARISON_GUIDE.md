# 🔄 Compare Two Datasets - Feature Guide

## Overview

The "Compare Two Datasets" feature allows you to compare a reference dataset (with declared labels) against a test dataset (without labels) to validate model accuracy and check consistency.

## Key Features

### ✅ Two Dataset Comparison
- **Reference Dataset**: Contains text with actual sentiment labels (declared/known)
- **Test Dataset**: Contains text without labels (non-declared, will be predicted)
- **Automatic Matching**: Matches texts between datasets for comparison
- **Numeric Filtering**: Automatically skips numeric-only entries

### ✅ Comprehensive Metrics
- Overall accuracy for matched samples
- Per-class metrics (Precision, Recall, F1-Score)
- Confusion matrix visualization
- Detailed comparison table
- Unmatched samples analysis

### ✅ Numeric Filtering
- **Automatic**: Skips numeric-only text entries
- **Smart Detection**: Identifies pure numbers, IDs, codes
- **User Feedback**: Shows count of filtered entries
- **No Manual Work**: Handles filtering automatically

## How to Use

### Step 1: Prepare Your Datasets

#### Reference Dataset (With Labels)
CSV format with text and sentiment columns:
```csv
text,sentiment
"I love this product!",Positive
"This is terrible.",Negative
"The product arrived on time.",Neutral
```

#### Test Dataset (Without Labels)
CSV format with text only:
```csv
text
"I love this product!"
"This is terrible."
"The product arrived on time."
```

### Step 2: Upload Datasets

1. Go to **"Compare Two Datasets"** mode
2. Upload **Reference Dataset** (with labels)
3. Upload **Test Dataset** (without labels)
4. The app will auto-detect columns

### Step 3: Select Columns

1. **Reference Dataset**:
   - Select text column
   - Select sentiment/label column

2. **Test Dataset**:
   - Select text column

### Step 4: Compare & Analyze

1. Click **"Compare Datasets & Calculate Accuracy"**
2. The app will:
   - Filter out numeric-only entries
   - Predict sentiments for test dataset
   - Match texts between datasets
   - Calculate accuracy metrics
   - Display results

### Step 5: Review Results

View:
- **Summary**: Total samples, matched, unmatched, skipped
- **Accuracy Metrics**: Overall accuracy for matched samples
- **Per-Class Metrics**: Precision, Recall, F1-Score
- **Confusion Matrix**: Visual representation
- **Detailed Comparison**: Side-by-side comparison table
- **Unmatched Samples**: New texts not in reference dataset

## Numeric Filtering

### What Gets Filtered

The app automatically skips:
- Pure numbers: `123`, `456.78`, `-100`
- Numeric IDs: `ID12345`, `SKU-789`
- Codes: `ABC123`, `123-456-789`
- Dates as numbers: `20240101`
- Any text that's >80% digits

### What Gets Analyzed

The app analyzes:
- Text with letters: `"I love this!"`
- Mixed content: `"Product ID: 123 is great"`
- Sentences: `"The price is $50"`
- Reviews: `"5 stars, excellent quality"`

### User Feedback

The app shows:
- Count of filtered entries
- Count of valid entries
- Clear messages about what was filtered

## Matching Logic

### Exact Match
- Texts are matched by exact comparison (case-insensitive)
- Whitespace is normalized
- Best for identical datasets

### Fuzzy Match (Future)
- Word overlap similarity
- 70%+ similarity threshold
- Handles minor variations

## Understanding Results

### Matched Samples
- Texts found in both datasets
- Used for accuracy calculation
- Shows predicted vs actual comparison

### Unmatched Samples
- Texts only in test dataset
- Shows predictions for new data
- Useful for discovering new patterns

### Skipped Entries
- Numeric-only entries filtered out
- Not suitable for sentiment analysis
- Count shown for transparency

## Use Cases

### 1. Model Validation
- Test model on new dataset
- Compare with known labels
- Measure accuracy on real data

### 2. Data Quality Check
- Verify dataset consistency
- Find discrepancies
- Identify data issues

### 3. Cross-Dataset Analysis
- Compare different data sources
- Check model generalization
- Validate across domains

### 4. Production Monitoring
- Compare production data with test set
- Monitor model performance
- Track accuracy over time

## Best Practices

### Dataset Preparation
1. **Reference Dataset**: Ensure labels are accurate
2. **Test Dataset**: Use similar data distribution
3. **Text Format**: Consistent formatting helps matching
4. **Size**: Larger datasets give more reliable metrics

### Label Format
- Use: "Positive", "Negative", "Neutral"
- Case-insensitive (auto-normalized)
- Consistent across datasets

### Text Matching
- **Exact Match**: Best for identical datasets
- **Similar Texts**: May not match if formatting differs
- **New Texts**: Will appear as unmatched

## Example Workflow

1. **Prepare Datasets**:
   - Reference: 100 labeled samples
   - Test: 150 unlabeled samples

2. **Upload & Compare**:
   - Upload both datasets
   - Select columns
   - Click "Compare"

3. **Review Results**:
   - 80 matched samples → Calculate accuracy
   - 70 unmatched samples → View predictions
   - 10 skipped (numeric) → Shown in summary

4. **Analyze**:
   - Check accuracy for matched samples
   - Review unmatched predictions
   - Identify patterns

## Troubleshooting

### No Matches Found
**Problem**: No texts matched between datasets

**Solutions**:
- Check text formatting differences
- Ensure similar text content
- Verify column selection
- Try exact text matching

### Low Accuracy
**Problem**: Accuracy is lower than expected

**Solutions**:
- Verify reference labels are correct
- Check for label inconsistencies
- Review confusion matrix
- Consider model retraining

### Many Unmatched Samples
**Problem**: Most test samples don't match reference

**Solutions**:
- This is normal if datasets are different
- Review unmatched predictions
- Use for discovering new patterns
- Consider separate analysis

## CSV Format Examples

### Reference Dataset
```csv
id,review_text,actual_sentiment
1,"I love this product!",Positive
2,"This is terrible.",Negative
3,"It's okay.",Neutral
```

### Test Dataset
```csv
id,review_text
1,"I love this product!"
2,"This is terrible."
3,"It's okay."
4,"New review text here."
```

## Tips

1. **Start Small**: Test with 50-100 samples first
2. **Check Matches**: Review matched pairs for accuracy
3. **Analyze Unmatched**: Learn from new predictions
4. **Monitor Filtering**: Check numeric filtering counts
5. **Compare Regularly**: Track accuracy over time

## Limitations

- Matching requires similar text content
- Numeric-only entries are automatically skipped
- Accuracy only calculated for matched samples
- Large datasets may take time to process

## Next Steps

After comparison:
1. **If accuracy is good**: Model is performing well
2. **If accuracy is low**: Review and improve model
3. **If many unmatched**: Analyze new patterns
4. **If filtering issues**: Check data quality

---

**Use Compare Two Datasets to validate your model across different datasets! 🔄📊**

