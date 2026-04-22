# 📊 Sentiment Analysis App

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://sentiment-analysis-mckv.streamlit.app/)
<<<<<<< HEAD
=======

>>>>>>> a37cf06c91543abd792fcf29d4392e259c76aca5

A comprehensive sentiment analysis application with **transformer-based models** (RoBERTa), word clouds, time-series analysis, and RESTful API.

## ✨ Features

- **Five Analysis Modes**: Dataset, Social Media (Twitter/YouTube), Image/Screenshot, Accuracy Meter/Validation, Manual Input
- **Transformer Model**: RoBERTa-base sentiment analysis (state-of-the-art accuracy)
- **Automatic Fallback**: Uses joblib model if transformer unavailable
- **Advanced Visualizations**: Word clouds, time-series charts, confidence scores
- **Progress Tracking**: Real-time progress bars for batch operations
- **OCR Support**: Extract text from images and screenshots
- **Export Results**: Download analysis as CSV

### 📱 Social Media Analysis

**Note**: Direct social media scraping has limitations on cloud platforms. The app provides:
- ✅ **CSV Upload**: Export social media data and upload as CSV (Recommended)
- ✅ **Manual Input**: Copy and paste text for analysis
- ✅ **Dataset Mode**: Best for bulk analysis

See `SOCIAL_MEDIA_SCRAPING_GUIDE.md` for detailed instructions.

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

App opens at `http://localhost:8501`

## 📁 Project Structure

```
├── app.py                 # Streamlit application (main file)
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
├── README-deploy.md       # Deployment guide
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

## 📚 Documentation

- **IMPROVEMENTS.md**: 50+ enhancement ideas and roadmap
- **README-deploy.md**: Deployment instructions
- **API Docs**: Auto-generated at `/docs` endpoint

## 🎯 Usage Examples

### Streamlit UI
- Upload CSV files with text data
- Analyze social media links
- Upload images/screenshots (OCR text extraction)
- Validate model accuracy with reference datasets
- Enter text manually
- View word clouds and time-series charts
- Download results
- **Automatic numeric filtering**: Skips numeric-only entries

## 🚀 Deployment & Hosting

### Quick Deploy (Streamlit Community Cloud - Recommended)

Your repository is ready for deployment at: **https://github.com/Nirotyay1302/sentiment-analysis**

1. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Repository: `Nirotyay1302/sentiment-analysis`
   - Branch: `main`
   - Main file: `app.py`
   - Click "Deploy!"
   - Your app will be live in 5-10 minutes!

**That's it!** Your app is now live with:
- ✅ Automatic HTTPS
- ✅ Free hosting
- ✅ Auto-updates on git push
- ✅ State-of-the-art transformer model

### Detailed Deployment Guide

See `DEPLOYMENT_GUIDE.md` for complete step-by-step instructions.

### Other Deployment Options

See `README-deploy.md` for detailed guides on:
- **Streamlit Community Cloud** (easiest, recommended)
- **Docker** (for containerized deployments)
- **VPS/Cloud VM** (traditional hosting)
- **Railway, Heroku, AWS** (cloud platforms)

## 🧪 Model

- **Primary Model**: RoBERTa-base Transformer (fine-tuned for Twitter sentiment)
  - Automatically downloads on first use (~500MB)
  - High accuracy for social media text
  - Pre-trained on large datasets
- **Fallback Model**: XGBoost/Logistic Regression (if transformer unavailable)
  - Train your own model using `train_model.py`
  - Test with `test_model.py`
- **Classes**: Positive, Neutral, Negative
- **Features**: Automatic model loading, confidence scores, probability breakdowns

### Training Your Own Model

```bash
# Train a custom model
python train_model.py

# Run tests
python test_model.py
```

See `TRAINING_RESULTS.md` for training details and test results.

## 🤝 Contributing

Contributions welcome! See `IMPROVEMENTS.md` for ideas.

## 📝 License

Open source and available for use.
