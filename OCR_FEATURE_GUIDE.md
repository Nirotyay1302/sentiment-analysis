# 📸 OCR (Optical Character Recognition) Feature Guide

## Overview

The Sentiment Analysis App now includes OCR functionality to extract text from images and screenshots, then analyze the sentiment of the extracted text.

## What is OCR?

OCR (Optical Character Recognition) is a technology that extracts text from images. This allows you to:
- Upload screenshots of comments, reviews, or social media posts
- Extract text from images automatically
- Analyze sentiment of the extracted text

## Features

### ✅ Supported Image Formats
- PNG
- JPG/JPEG
- WEBP

### ✅ Use Cases
- **Social Media Screenshots**: Analyze comments from Twitter, Facebook, Instagram, etc.
- **Review Screenshots**: Extract and analyze product/service reviews
- **Comment Sections**: Analyze YouTube, Reddit, or forum comment screenshots
- **Text in Images**: Any image containing readable text

## How to Use

### Step 1: Select Mode
1. Go to the sidebar
2. Select **"Analyze Image/Screenshot"** from the mode dropdown

### Step 2: Upload Image
1. Click "Upload an image or screenshot"
2. Select an image file (PNG, JPG, JPEG, or WEBP)
3. The image will be displayed

### Step 3: Extract Text & Analyze
1. Click **"Extract Text & Analyze Sentiment"** button
2. Wait for OCR processing (30-60 seconds on first use)
3. View extracted text
4. See sentiment analysis results

### Step 4: Review Results
- **Extracted Text**: View all text segments found in the image
- **Sentiment Analysis**: See overall sentiment (Positive, Neutral, Negative)
- **Confidence Score**: Check prediction confidence
- **Segment Analysis**: Analyze individual text segments (if multiple)
- **Download Results**: Save results as CSV

## Technical Details

### OCR Library
- **Library**: EasyOCR
- **Language**: English (en)
- **GPU**: Disabled (CPU-only for cloud compatibility)
- **Model Size**: ~100MB (downloaded on first use)

### Performance
- **First Use**: 30-60 seconds (downloads models)
- **Subsequent Uses**: 5-15 seconds per image
- **Accuracy**: Depends on image quality and text clarity

### Requirements
- `easyocr>=1.7.0`
- `Pillow>=9.0.0`
- Internet connection (for first-time model download)

## Best Practices

### For Best Results

1. **Image Quality**:
   - Use high-resolution images
   - Ensure text is clear and readable
   - Good contrast between text and background

2. **Text Clarity**:
   - Avoid blurry or distorted text
   - Use images with standard fonts
   - Avoid decorative or stylized fonts

3. **Image Format**:
   - PNG is preferred for screenshots
   - JPG/JPEG works well for photos
   - WEBP is supported but less common

### Common Issues

#### No Text Extracted
- **Problem**: OCR couldn't find text in the image
- **Solutions**:
  - Ensure the image contains visible text
  - Try a higher resolution image
  - Check image clarity and contrast
  - Use "Manual Text Input" mode as alternative

#### Low Accuracy
- **Problem**: Extracted text has errors
- **Solutions**:
  - Use clearer images
  - Ensure good lighting/contrast
  - Avoid handwritten text (OCR works best with printed text)
  - Review and correct extracted text manually

#### Slow Processing
- **Problem**: OCR takes too long
- **Solutions**:
  - First use is slower (model download)
  - Subsequent uses are faster
  - Reduce image size if very large
  - Be patient on first use

## Streamlit Cloud Compatibility

### ✅ Fully Compatible
- EasyOCR works on Streamlit Cloud
- No additional system dependencies required
- Models download automatically
- CPU-only mode (GPU not needed)

### Limitations
- First-time model download takes 30-60 seconds
- Processing is slower than local GPU setup
- Memory usage: ~500MB for OCR models

## Alternatives

If OCR doesn't work or you prefer manual input:

1. **Manual Text Input Mode**:
   - Copy text from images manually
   - Paste into "Manual Text Input" mode
   - Get instant sentiment analysis

2. **CSV Upload**:
   - Export text data to CSV
   - Use "Analyze Dataset" mode
   - Bulk analysis of multiple texts

## Troubleshooting

### OCR Not Available
**Error**: "OCR (Optical Character Recognition) is not available"

**Solution**:
1. Install required packages:
   ```bash
   pip install easyocr Pillow
   ```
2. Restart the app
3. Try again

### Model Download Fails
**Error**: "Failed to initialize OCR reader"

**Solution**:
1. Check internet connection
2. Ensure sufficient disk space (~500MB)
3. Try again (may be temporary network issue)
4. Check Streamlit Cloud logs

### Text Extraction Errors
**Error**: "Error extracting text from image"

**Solution**:
1. Verify image format is supported
2. Check image file is not corrupted
3. Try a different image
4. Use "Manual Text Input" mode as fallback

## Examples

### Example 1: Social Media Screenshot
1. Take a screenshot of Twitter comments
2. Upload to app
3. Extract all comments
4. Analyze sentiment of each comment

### Example 2: Review Screenshot
1. Screenshot product reviews from Amazon
2. Upload to app
3. Extract review text
4. Get overall sentiment analysis

### Example 3: Comment Section
1. Screenshot YouTube comment section
2. Upload to app
3. Extract all comments
4. Analyze sentiment distribution

## Performance Tips

1. **Image Size**: Smaller images process faster
2. **Text Density**: Images with more text take longer
3. **First Use**: Be patient on first use (model download)
4. **Caching**: OCR models are cached after first use

## Security & Privacy

- Images are processed in memory
- No images are stored permanently
- Text extraction happens server-side
- Results can be downloaded as CSV

## Future Enhancements

Potential improvements:
- Support for multiple languages
- Handwritten text recognition
- Better accuracy with training
- Batch image processing
- Real-time OCR preview

## Support

If you encounter issues:
1. Check this guide
2. Review troubleshooting section
3. Check app logs
4. Try alternative methods (Manual Input, CSV Upload)

---

**Enjoy using OCR for sentiment analysis! 📸✨**

