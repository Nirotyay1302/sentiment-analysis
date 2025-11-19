@echo off
echo ========================================
echo   Sentiment Analysis App - Starting
echo ========================================
echo.
echo Starting Streamlit app...
echo The app will open in your browser automatically.
echo If not, visit: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app.
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause

