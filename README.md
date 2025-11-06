# 📊 Sentiment Analysis App

A comprehensive sentiment analysis application with **transformer-based models** (RoBERTa), word clouds, time-series analysis, and RESTful API.

## ✨ Features

- **Three Analysis Modes**: Dataset, Social Media (Twitter/YouTube), Manual Input
- **Transformer Model**: RoBERTa-base sentiment analysis (state-of-the-art accuracy)
- **Automatic Fallback**: Uses joblib model if transformer unavailable
- **Advanced Visualizations**: Word clouds, time-series charts, confidence scores
- **RESTful API**: FastAPI server for programmatic access
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

# Train model with included data
python train_model.py --input data/training_data.csv

# Run Streamlit app
streamlit run app.py
```

App opens at `http://localhost:8501`

### API Server

```bash
# In another terminal
uvicorn api:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

## 📁 Project Structure

```
├── app.py                 # Streamlit application
├── api.py                 # FastAPI REST API
├── train_model.py         # Model training script
├── model.joblib           # Trained XGBoost model
├── data/
│   └── training_data.csv  # Training dataset
├── IMPROVEMENTS.md        # Future enhancements
├── README-deploy.md       # Deployment guide
└── requirements.txt       # Dependencies
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

### API Usage
```bash
# Single text analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'

# Batch analysis
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible!", "OK"]}'
```

## 🚀 Deployment & Hosting

### Quick Deploy (Streamlit Community Cloud - Recommended)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app" → Select your repo → Main file: `app.py`
   - Click "Deploy!"
   - Your app will be live in 5-10 minutes!

**That's it!** Your app is now live with:
- ✅ Automatic HTTPS
- ✅ Free hosting
- ✅ Auto-updates on git push
- ✅ State-of-the-art transformer model

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
- **Classes**: Positive, Neutral, Negative
- **Features**: Automatic model loading, confidence scores, probability breakdowns

## 🤝 Contributing

Contributions welcome! See `IMPROVEMENTS.md` for ideas.

## 📝 License

Open source and available for use.
