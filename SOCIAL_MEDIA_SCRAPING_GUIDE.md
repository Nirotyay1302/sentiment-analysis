# 📱 Social Media Scraping Guide

## Understanding the Limitations

### Why Social Media Scraping May Not Work

1. **Platform Restrictions**: 
   - Twitter/X and YouTube have strict API policies
   - Direct scraping violates terms of service
   - Cloud platforms (like Streamlit Cloud) block scraping attempts

2. **Library Compatibility**:
   - `snscrape` is archived and incompatible with Python 3.13+
   - Streamlit Cloud uses Python 3.13, so Twitter scraping won't work
   - `youtube-comment-downloader` may have compatibility issues

3. **API Requirements**:
   - Official APIs require authentication and have rate limits
   - Free tiers have restrictions
   - Setup requires API keys and configuration

## ✅ Recommended Alternatives

### Option 1: CSV Upload (Recommended)

**Best for**: Bulk analysis, reliable, no API needed

1. **Export your social media data**:
   - Twitter: Use Twitter's data export feature or third-party tools
   - YouTube: Use browser extensions or YouTube Data API
   - Export as CSV with a "text" or "comment" column

2. **Upload to the app**:
   - Go to "Analyze Dataset" mode
   - Upload your CSV file
   - Select the text column
   - Analyze!

### Option 2: Manual Text Input

**Best for**: Quick analysis of individual posts/comments

1. Copy text from social media posts
2. Go to "Manual Text Input" mode
3. Paste the text
4. Get instant sentiment analysis

### Option 3: Use Official APIs (Advanced)

**Best for**: Production applications, automated analysis

#### Twitter/X API
- Sign up for Twitter API access
- Use `tweepy` library
- Requires API keys and authentication
- Has rate limits and costs

#### YouTube Data API
- Get YouTube Data API key from Google Cloud
- Use `google-api-python-client` library
- Requires API key setup
- Has quota limits

## 📊 How to Export Social Media Data

### Twitter/X Data Export

1. **Official Twitter Export**:
   - Go to Twitter Settings → Your Account → Download an archive
   - Request your data archive
   - Wait for email confirmation
   - Download and extract the archive
   - Convert to CSV format

2. **Third-party Tools**:
   - Use browser extensions like "Twitter Archive Eraser" or "Tweet Exporter"
   - Export tweets to CSV format
   - Ensure CSV has a "text" column

### YouTube Comments Export

1. **Browser Extensions**:
   - Install "YouTube Comment Exporter" extension
   - Go to any YouTube video
   - Click the extension icon
   - Export comments to CSV

2. **YouTube Data API**:
   - Get API key from Google Cloud Console
   - Use Python script to fetch comments
   - Save to CSV format

3. **Manual Copy-Paste**:
   - Copy comments from YouTube
   - Paste into a text file or spreadsheet
   - Convert to CSV with a "comment" column

## 🔧 For Local Development

If you want to test social media scraping locally:

### Twitter Scraping (Local Only)

```bash
# Install snscrape (only works on Python < 3.13)
pip install snscrape

# Note: This won't work on Streamlit Cloud (Python 3.13)
```

### YouTube Comments (Local Only)

```bash
# Install youtube-comment-downloader
pip install youtube-comment-downloader

# Note: May have compatibility issues on cloud platforms
```

## 💡 Best Practices

1. **Use CSV Upload**: Most reliable method
2. **Respect Terms of Service**: Don't violate platform policies
3. **Use Official APIs**: For production applications
4. **Manual Input**: For quick, one-off analyses
5. **Dataset Mode**: Best for bulk analysis

## 🚫 What Won't Work

- ❌ Direct scraping on Streamlit Cloud (blocked by platform)
- ❌ snscrape on Python 3.13+ (library incompatible)
- ❌ Unauthorized API access (violates terms of service)
- ❌ Bypassing rate limits (may result in bans)

## ✅ What Will Work

- ✅ CSV upload (any platform)
- ✅ Manual text input (any platform)
- ✅ Official API integration (with proper setup)
- ✅ Dataset analysis (bulk processing)

## 📝 Example CSV Format

Your CSV file should have this structure:

```csv
text,sentiment
"I love this product!",Positive
"This is terrible.",Negative
"The product arrived on time.",Neutral
```

Or simply:

```csv
text
"I love this product!"
"This is terrible."
"The product arrived on time."
```

The app will automatically detect the text column and analyze it.

## 🆘 Need Help?

1. Check the app's "Analyze Dataset" mode for CSV upload
2. Use "Manual Text Input" for quick analysis
3. Review this guide for export instructions
4. Check the main README for general usage

---

**Remember**: Social media scraping is optional. The app works perfectly with CSV uploads and manual input! 🎉

