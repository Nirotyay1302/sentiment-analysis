# 🚀 Deployment Guide

This guide covers multiple deployment options for the Sentiment Analysis App.

## Option 1: Streamlit Community Cloud (Recommended - Easiest) ⭐

### Prerequisites
- GitHub account
- Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Steps

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set **Main file path** to: `app.py`
   - Set **App URL** (optional)
   - Click "Deploy!"

3. **Wait for deployment**
   - First deployment may take 5-10 minutes (downloads transformer model ~500MB)
   - Subsequent deployments are faster

4. **Access your app**
   - Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Important Notes for Streamlit Community Cloud
- ✅ No model.joblib needed (uses transformer model automatically)
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Auto-updates on git push
- ⚠️ First model download may take time
- ⚠️ Free tier has resource limits

---

## Option 2: Docker Deployment

### Build and Run Locally

```bash
# Build Docker image
docker build -t sentiment-app:latest .

# Run container
docker run -p 8501:8501 sentiment-app:latest
```

### Deploy to Cloud Platforms

#### Docker Hub
```bash
# Tag image
docker tag sentiment-app:latest YOUR_DOCKERHUB_USERNAME/sentiment-app:latest

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/sentiment-app:latest
```

#### AWS ECS / Google Cloud Run / Azure Container Instances
- Use the Docker image with any container platform
- Set port mapping to 8501
- Configure environment variables if needed

---

## Option 3: Traditional VPS/Cloud VM

### Requirements
- Ubuntu/Debian server (or similar)
- Python 3.11+
- 2GB+ RAM recommended (for transformer model)

### Setup Steps

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run with Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### Production with Systemd (Linux)

Create `/etc/systemd/system/sentiment-app.service`:

```ini
[Unit]
Description=Sentiment Analysis App
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/your/app
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sentiment-app
sudo systemctl start sentiment-app
```

### With Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

---

## Option 4: Heroku

### Prerequisites
- Heroku CLI installed
- Heroku account

### Steps

1. **Create `Procfile`**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**
   ```bash
   heroku create your-app-name
   git push heroku main
   heroku open
   ```

**Note**: Heroku free tier discontinued. Consider paid tier or alternatives.

---

## Option 5: Railway

1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects and deploys
6. Set environment variables if needed

---

## Environment Variables

You can set these if needed:

```bash
# Optional: Hugging Face token (for private models)
HUGGINGFACE_TOKEN=your_token_here

# Optional: Health check port
HEALTH_PORT=8502
```

---

## Troubleshooting

### Model Download Issues
- First run downloads ~500MB transformer model
- Ensure sufficient disk space
- Check internet connection

### Port Issues
- Ensure port 8501 is open (or use platform's port mapping)
- Check firewall settings

### Memory Issues
- Transformer model requires ~2GB RAM
- Consider upgrading server resources
- Use CPU-only PyTorch if needed

### Import Errors
- Ensure all dependencies in `requirements.txt` are installed
- Check Python version (3.11+ recommended)

---

## Performance Tips

1. **Caching**: Streamlit automatically caches model loading
2. **Model Size**: Consider using smaller models for limited resources
3. **Batch Processing**: Process texts in batches for better performance
4. **CDN**: Use CDN for static assets if deploying on VPS

---

## Security Checklist

- ✅ Use HTTPS (automatic on Streamlit Cloud)
- ✅ Don't commit secrets.toml
- ✅ Use environment variables for sensitive data
- ✅ Enable authentication if needed (Streamlit Cloud supports this)
- ✅ Regular dependency updates

---

## Support

For issues:
1. Check logs in your deployment platform
2. Review Streamlit documentation
3. Check GitHub issues

---

## Quick Deploy Commands

```bash
# Streamlit Community Cloud (just push to GitHub)
git push origin main

# Docker
docker build -t sentiment-app . && docker run -p 8501:8501 sentiment-app

# Local
streamlit run app.py
```
