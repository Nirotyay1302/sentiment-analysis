# 🚀 Quick Run Guide

## How to Run the App

### Option 1: Using Python Module (Recommended)
```bash
python -m streamlit run app.py
```

### Option 2: Using Batch File (Windows)
```bash
run_app.bat
```

### Option 3: Using PowerShell Script
```powershell
.\run_app.ps1
```

### Option 4: If Streamlit is in PATH
```bash
streamlit run app.py
```

## Troubleshooting

### ❌ "streamlit is not recognized"
**Solution**: Use `python -m streamlit run app.py` instead

### ❌ "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: Install dependencies:
```bash
python -m pip install -r requirements.txt
```

### ❌ "Port 8501 already in use"
**Solution**: Use a different port:
```bash
python -m streamlit run app.py --server.port=8502
```

## Access the App

Once running, open your browser to:
- **http://localhost:8501**
- The app will open automatically in most cases

## First Run Notes

- First time loading may take 1-2 minutes (downloads transformer model ~500MB)
- Model is cached after first download
- Subsequent runs are faster

## Stopping the App

Press `Ctrl+C` in the terminal to stop the app.

---

**That's it!** Your app should now be running! 🎉

