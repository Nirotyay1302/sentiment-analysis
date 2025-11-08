# 📊 Sentiment Analysis App

A comprehensive sentiment analysis application with **transformer-based models** (RoBERTa), word clouds, time-series analysis, and RESTful API.

## ✨ Features

- **Three Analysis Modes**: Dataset, Social Media (Twitter/YouTube), Manual Input
- **Transformer Model**: RoBERTa-base sentiment analysis (state-of-the-art accuracy)
- **Automatic Fallback**: Uses joblib model if transformer unavailable
- **Advanced Visualizations**: Word clouds, time-series charts, confidence scores
- **Progress Tracking**: Real-time progress bars for batch operations
- **Export Results**: Download analysis as CSV

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
- Enter text manually
- View word clouds and time-series charts
- Download results

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
