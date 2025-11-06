# 🚀 Quick Start Guide

Get your Sentiment Analysis App running in minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First installation may take a few minutes (downloads PyTorch and transformers).

## Step 2: Run the App Locally

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

## Step 3: Test the App

1. **Manual Text Input** (easiest test):
   - Select "Manual Text Input" from sidebar
   - Enter: "I love this product!"
   - Click "Analyze Text"
   - You should see: **Positive** sentiment with high confidence

2. **Try another example**:
   - Enter: "This is terrible and disappointing"
   - Should show: **Negative** sentiment

## Step 4: Deploy to the Cloud

### Option A: Streamlit Community Cloud (5 minutes) ⭐

1. **Create GitHub repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Push to GitHub**:
   - Create new repo on GitHub
   - Follow GitHub's instructions to push

3. **Deploy to Streamlit**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `app.py`
   - Click "Deploy!"

4. **Wait 5-10 minutes**:
   - First deployment downloads transformer model (~500MB)
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Option B: Docker

```bash
# Build image
docker build -t sentiment-app .

# Run container
docker run -p 8501:8501 sentiment-app
```

### Option C: Your Own Server

```bash
# On your server
git clone YOUR_REPO_URL
cd YOUR_REPO
pip install -r requirements.txt
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

## Troubleshooting

### ❌ "Import transformers not found"
**Solution**: Run `pip install -r requirements.txt`

### ❌ "Model download fails"
**Solution**: 
- Check internet connection
- Ensure ~1GB free disk space
- First download takes time (~500MB model)

### ❌ "App not loading"
**Solution**:
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port=8502`

### ❌ "Out of memory"
**Solution**:
- Transformer model needs ~2GB RAM
- Close other applications
- Consider using CPU-only PyTorch

## Next Steps

1. ✅ Test the app locally
2. ✅ Deploy to Streamlit Cloud
3. ✅ Share your app URL
4. ✅ Analyze your data!

## Features to Try

- **Dataset Analysis**: Upload a CSV with text column
- **Social Media**: Analyze Twitter or YouTube comments
- **Manual Input**: Test individual sentences
- **Visualizations**: View word clouds, charts, and statistics

## Need Help?

- Check `README-deploy.md` for detailed deployment guides
- Review `README.md` for full documentation
- Check logs in Streamlit Cloud dashboard

---

**You're all set!** 🎉

Your sentiment analysis app is ready to use!

