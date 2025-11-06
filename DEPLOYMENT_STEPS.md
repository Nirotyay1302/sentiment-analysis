# 📋 Step-by-Step Deployment Guide

Follow these steps to host your Sentiment Analysis App!

## 🎯 Recommended: Streamlit Community Cloud

### Step 1: Prepare Your Code

✅ All files are ready:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Dependencies
- ✅ `.streamlit/config.toml` - Streamlit config
- ✅ `.gitignore` - Git ignore rules
- ✅ `README-deploy.md` - Detailed deployment guide

### Step 2: Initialize Git Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Sentiment Analysis App with Transformer Model"
```

### Step 3: Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click "+" → "New repository"
3. Name it (e.g., `sentiment-analysis-app`)
4. **Don't** initialize with README (we already have one)
5. Click "Create repository"

### Step 4: Push to GitHub

```bash
# Add remote (replace with your username and repo name)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note**: If you haven't set up Git, you'll need to:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 5: Deploy to Streamlit Cloud

1. **Go to Streamlit Community Cloud**
   - Visit: [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Or click "Get started" if first time

3. **Configure App**
   - **Repository**: Select your repository from dropdown
   - **Branch**: `main` (or `master`)
   - **Main file path**: `app.py`
   - **App URL**: Choose a unique name (e.g., `my-sentiment-app`)
     - Will be: `https://my-sentiment-app.streamlit.app`

4. **Deploy!**
   - Click "Deploy!" button
   - Wait 5-10 minutes for first deployment
   - You'll see a progress bar showing:
     - Installing dependencies
     - Downloading transformer model (~500MB)
     - Starting app

5. **Access Your App**
   - Once deployed, click "Manage app" → "Open app"
   - Or visit: `https://YOUR_APP_NAME.streamlit.app`

### Step 6: Test Your Deployed App

1. ✅ Test "Manual Text Input" mode
2. ✅ Try uploading a CSV file
3. ✅ Test social media link analysis
4. ✅ Check visualizations

---

## 🚀 Alternative: Docker Deployment

### Step 1: Build Docker Image

```bash
docker build -t sentiment-app:latest .
```

### Step 2: Run Container

```bash
docker run -p 8501:8501 sentiment-app:latest
```

### Step 3: Access App

Open browser: `http://localhost:8501`

### Step 4: Deploy to Cloud (Docker)

**AWS ECS / Google Cloud Run / Azure Container Instances:**

1. Push image to container registry:
   ```bash
   # Tag image
   docker tag sentiment-app:latest YOUR_REGISTRY/sentiment-app:latest
   
   # Push
   docker push YOUR_REGISTRY/sentiment-app:latest
   ```

2. Deploy using your cloud platform's container service

---

## 📊 Deployment Checklist

Before deploying, verify:

- [ ] All files committed to git
- [ ] `requirements.txt` includes all dependencies
- [ ] `.streamlit/config.toml` exists
- [ ] `.gitignore` includes secrets
- [ ] App runs locally (`streamlit run app.py`)
- [ ] No errors in terminal

---

## 🔧 Post-Deployment

### Update Your App

1. Make changes locally
2. Commit and push:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push origin main
   ```
3. Streamlit Cloud automatically redeploys!

### Monitor Your App

- Check Streamlit Cloud dashboard for:
  - App status
  - Logs
  - Resource usage
  - Error messages

### Custom Domain (Optional)

1. Go to Streamlit Cloud dashboard
2. Click "Settings"
3. Add custom domain (requires DNS setup)

---

## 🆘 Troubleshooting

### Deployment Fails

**Error**: "ModuleNotFoundError"
- **Solution**: Check `requirements.txt` includes all dependencies

**Error**: "Model download fails"
- **Solution**: Check internet connection, ensure sufficient disk space

**Error**: "Port already in use"
- **Solution**: Change port in Streamlit config or Dockerfile

### App Won't Start

1. Check logs in Streamlit Cloud dashboard
2. Verify all dependencies in `requirements.txt`
3. Test locally first: `streamlit run app.py`

### Slow Performance

- First load: Model download takes time (~500MB)
- Subsequent loads: Faster (model cached)
- Consider upgrading resources if needed

---

## 📚 Additional Resources

- **Streamlit Cloud Docs**: [docs.streamlit.io/streamlit-community-cloud](https://docs.streamlit.io/streamlit-community-cloud)
- **Detailed Deployment Guide**: See `README-deploy.md`
- **Quick Start**: See `QUICK_START.md`

---

## ✅ Success!

Your app should now be live! 🎉

Share your app URL and start analyzing sentiment!

