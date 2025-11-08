# 🚀 Deployment Guide - Streamlit Cloud

Complete guide to deploy your Sentiment Analysis App to Streamlit Cloud.

## Prerequisites

- ✅ GitHub account
- ✅ Code pushed to GitHub repository: `https://github.com/Nirotyay1302/sentiment-analysis.git`
- ✅ Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

## Step-by-Step Deployment

### Step 1: Verify Repository

Your repository should contain:
- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `README.md` - Documentation

### Step 2: Sign in to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in"** or **"Get started"**
3. Sign in with your **GitHub account**
4. Authorize Streamlit Cloud to access your repositories

### Step 3: Deploy Your App

1. Click **"New app"** button
2. Fill in the deployment form:

   **Repository**: Select `Nirotyay1302/sentiment-analysis`
   
   **Branch**: Select `main`
   
   **Main file path**: Enter `app.py`
   
   **App URL**: Choose a unique name (e.g., `sentiment-analysis-app`)
     - Your app will be live at: `https://sentiment-analysis-app.streamlit.app`

3. Click **"Deploy!"** button

### Step 4: Wait for Deployment

- **First deployment**: Takes 5-10 minutes
  - Downloads transformer model (~500MB)
  - Installs all dependencies
  - Builds the app environment

- **Subsequent deployments**: Faster (~2-3 minutes)
  - Only updates changed files

### Step 5: Access Your App

Once deployment completes:
- Click **"Manage app"** → **"Open app"**
- Or visit: `https://YOUR_APP_NAME.streamlit.app`

## 🎉 Success!

Your app is now live on Streamlit Cloud with:
- ✅ Automatic HTTPS
- ✅ Free hosting
- ✅ Auto-updates on git push
- ✅ State-of-the-art transformer model

## 📝 Updating Your App

To update your deployed app:

1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Your update message"
   git push origin main
   ```
3. Streamlit Cloud automatically redeploys!

## ⚠️ Important Notes

### Model Loading
- The transformer model downloads automatically on first use (~500MB)
- First model load may take 2-5 minutes
- Model is cached for subsequent uses

### Resource Limits (Free Tier)
- App timeout: 30 minutes of inactivity
- Memory: 1GB RAM
- CPU: Shared resources
- Storage: Sufficient for model cache

### Troubleshooting

#### App won't deploy
- ✅ Check that `app.py` exists in repository
- ✅ Verify `requirements.txt` is correct
- ✅ Check Streamlit Cloud logs for errors

#### Import errors
- ✅ Ensure all dependencies in `requirements.txt`
- ✅ Check Python version compatibility (3.11+)
- ✅ Verify package versions are compatible

#### Model download fails
- ✅ Check internet connection
- ✅ Ensure sufficient disk space (~1GB)
- ✅ First download takes time - be patient
- ✅ Check Streamlit Cloud logs

#### App crashes or times out
- ✅ Check memory usage
- ✅ Reduce batch processing size
- ✅ Optimize model loading
- ✅ Check Streamlit Cloud resource limits

## 🔗 Useful Links

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Cloud Status](https://status.streamlit.io/)
- [GitHub Repository](https://github.com/Nirotyay1302/sentiment-analysis)
- [Streamlit Community Forum](https://discuss.streamlit.io/)

## 📞 Support

If you encounter issues:
1. Check Streamlit Cloud logs in the dashboard
2. Review the troubleshooting section above
3. Visit [Streamlit Community Forum](https://discuss.streamlit.io/)
4. Check [GitHub Issues](https://github.com/Nirotyay1302/sentiment-analysis/issues)

---

**Happy Deploying! 🚀**

