# 🚀 Deploy to Streamlit Cloud - Step by Step

Follow these steps to deploy your Sentiment Analysis App to Streamlit Cloud.

## ✅ Prerequisites

- ✅ GitHub account
- ✅ Code pushed to GitHub repository
- ✅ Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

## 📋 Step-by-Step Deployment

### Step 1: Verify Your Repository

Your repository should have these essential files:
- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `README.md` - Documentation

### Step 2: Go to Streamlit Cloud

1. Visit: **https://share.streamlit.io**
2. Sign in with your **GitHub account**
3. Click **"New app"** button (or "Get started" if first time)

### Step 3: Configure Your App

Fill in the deployment form:

1. **Repository**: Select `Nirotyay1302/Sentiment-Analysis` from dropdown
2. **Branch**: Select `main`
3. **Main file path**: Enter `app.py`
4. **App URL**: Choose a unique name (e.g., `sentiment-analysis-app`)
   - Your app will be: `https://sentiment-analysis-app.streamlit.app`

### Step 4: Deploy!

1. Click **"Deploy!"** button
2. Wait 5-10 minutes for first deployment
   - Downloads transformer model (~500MB)
   - Installs dependencies
   - Starts the app

### Step 5: Access Your App

Once deployment completes:
- Click **"Manage app"** → **"Open app"**
- Or visit: `https://YOUR_APP_NAME.streamlit.app`

## 🎉 Done!

Your app is now live on Streamlit Cloud!

## 📝 Future Updates

To update your app:
1. Make changes locally
2. Commit and push:
   ```bash
   git add .
   git commit -m "Your update message"
   git push origin main
   ```
3. Streamlit Cloud automatically redeploys!

## ⚠️ Important Notes

- **First deployment**: Takes 5-10 minutes (model download)
- **Subsequent deployments**: Faster (~2-3 minutes)
- **Auto-updates**: App redeploys automatically on git push
- **Free tier**: Includes resource limits
- **HTTPS**: Automatic and free

## 🆘 Troubleshooting

### App won't deploy
- Check that `app.py` exists in repository
- Verify `requirements.txt` is correct
- Check Streamlit Cloud logs for errors

### Import errors
- Ensure all dependencies in `requirements.txt`
- Check Python version compatibility

### Model download fails
- Check internet connection
- Ensure sufficient disk space (~1GB)
- First download takes time - be patient

---

**Need help?** Check the logs in Streamlit Cloud dashboard or see `README-deploy.md` for more details.

