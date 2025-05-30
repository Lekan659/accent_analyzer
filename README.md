#Accent Analyzer

A working tool that analyzes English accents from video/audio using Gemini AI for direct voice pattern analysis.

## 📋 Requirements

Create `requirements.txt`:

```
streamlit==1.29.0
anthropic==0.8.1
librosa==0.10.1
soundfile==0.12.1
yt-dlp==2023.12.30
numpy==1.24.3
requests==2.31.0
```

## 🔧 Setup Instructions

### 1. Get Gemini API Key
```bash
# Get your key from: https://console.anthropic.com/
export GOOGLE_GEMINI_API_KEY="your-actual-key-here"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
streamlit run main.py
```

### 4. Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Add `GOOGLE_GEMINI_API_KEY` in App Settings > Secrets
5. Deploy!

## 🎯 How It Works

1. **Audio Extraction**: Uses yt-dlp to extract audio from any video URL
2. **Feature Analysis**: Librosa extracts acoustic features (formants, MFCC, spectral data)
3. **Gemini AI Analysis**: Sends features to Gemini for direct accent pattern recognition
4. **Results**: Returns accent classification with confidence score

## 🔍 Key Technical Features

- **Direct Voice Analysis**: No text transcription - analyzes audio patterns directly
- **Multiple Input Methods**: Video URLs (Loom, YouTube, etc.) or file uploads
- **Acoustic Feature Extraction**: Formant frequencies, MFCC, spectral analysis
- **Gemini AI Integration**: Uses Gemini
