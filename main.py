import streamlit as st
import librosa
import numpy as np
import os
import tempfile
import json
import yt_dlp
import google.generativeai as genaigoogle
from google import genai
from scipy.signal import find_peaks
import requests
from urllib.parse import urlparse
import subprocess

# Page config
st.set_page_config(
    page_title="English Accent Detector",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 English Accent Detection Tool")
st.markdown("*Built for REM Waste*")

# Initialize Gemini
@st.cache_resource
def init_gemini():
    try:
        # Try multiple ways to get the API key
        api_key = "AIzaSyCHoRRI4L2Y1KyrjW4LIQ28B14ZVVQWszM"
        # if "GOOGLE_GEMINI_API_KEY" in st.secrets:
        #     api_key = st.secrets["GOOGLE_GEMINI_API_KEY"]
        # elif "google_gemini_api_key" in st.secrets:
        #     api_key = st.secrets["google_gemini_api_key"]
        # elif os.getenv("GOOGLE_GEMINI_API_KEY"):
        #     api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        
        if not api_key:
            return False, "Google Gemini API key not found. Please set it in Streamlit secrets."
        
        genaigoogle.configure(api_key=api_key)
        # Test the connection
        model = genaigoogle.GenerativeModel('gemini-1.5-flash')
        test_response = model.generate_content("Hello")
        return True, "Gemini API configured successfully!"
    except Exception as e:
        return False, f"Error configuring Gemini: {str(e)}"

# Check FFmpeg availability
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except:
        return False

def estimate_formants(y, sr):
    """Estimate formant frequencies"""
    try:
        n_fft = 2048
        D = librosa.stft(y, n_fft=n_fft, hop_length=n_fft//4, window='hann')
        magnitudes = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        avg_magnitude = np.mean(magnitudes, axis=1)
        
        # Focus on speech formant range
        relevant_indices = np.where((freqs > 200) & (freqs < 4000))[0]
        
        if len(relevant_indices) == 0:
            return [500, 1500, 2500, 3500]
        
        # Find peaks
        peaks, _ = find_peaks(
            avg_magnitude[relevant_indices], 
            height=np.max(avg_magnitude[relevant_indices]) * 0.1,
            distance=20
        )
        
        if len(peaks) == 0:
            return [500, 1500, 2500, 3500]
        
        formant_freqs = sorted([float(freqs[relevant_indices[p]]) for p in peaks if freqs[relevant_indices[p]] > 200])
        
        # Return first 4 formants, pad with defaults if needed
        defaults = [500, 1500, 2500, 3500]
        while len(formant_freqs) < 4:
            formant_freqs.append(defaults[len(formant_freqs)])
        
        return formant_freqs[:4]
        
    except Exception as e:
        st.warning(f"Formant estimation error: {e}")
        return [500, 1500, 2500, 3500]

def extract_audio_features(y, sr):
    """Extract comprehensive audio features"""
    try:
        features = {}
        
        # Basic spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = [float(x) for x in np.mean(mfcc, axis=1)]
        
        # Tempo and rhythm
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
        except:
            features['tempo'] = 120.0
        
        # Pitch/F0 analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        features['mean_pitch'] = float(np.mean(pitch_values)) if pitch_values else 0.0
        features['pitch_std'] = float(np.std(pitch_values)) if pitch_values else 0.0
        
        # Formants
        features['formants'] = estimate_formants(y, sr)
        
        # Additional prosodic features
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
        features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
        
        return features
        
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

def download_audio_from_url(url):
    """Download and extract audio from various URL types"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Check if it's a direct audio/video file
        parsed_url = urlparse(url)
        if parsed_url.path.endswith(('.mp3', '.wav', '.m4a', '.ogg')):
            # Direct audio file
            response = requests.get(url, stream=True)
            audio_path = os.path.join(temp_dir, "audio.mp3")
            with open(audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return audio_path, temp_dir
        
        # Use yt-dlp for video URLs
        audio_path = os.path.join(temp_dir, "extracted_audio.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Find the downloaded file
        final_audio_path = os.path.join(temp_dir, "extracted_audio.mp3")
        if not os.path.exists(final_audio_path):
            # Look for any audio file in temp dir
            audio_files = [f for f in os.listdir(temp_dir) if f.endswith(('.mp3', '.wav', '.m4a'))]
            if audio_files:
                final_audio_path = os.path.join(temp_dir, audio_files[0])
            else:
                raise Exception("No audio file found after extraction")
        
        return final_audio_path, temp_dir
        
    except Exception as e:
        # Cleanup on error
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        except:
            pass
        raise e

def analyze_with_gemini(features, duration, audio_path):
    """Analyze features with Gemini AI"""

    client = genai.Client(api_key="AIzaSyCHoRRI4L2Y1KyrjW4LIQ28B14ZVVQWszM")

    myfile = client.files.upload(file=audio_path)
    
    # Format features for the prompt
    mfcc_str = ", ".join([f"{x:.2f}" for x in features['mfcc_mean'][:5]])
    formants_str = ", ".join([f"{x:.0f}Hz" for x in features['formants']])
    
    prompt = f"""
You are an expert linguist specializing in English accent classification. Analyze the audio clip and then decide the accent from a {duration:.1f}-second English speech sample:



TASK: Classify the English accent and provide confidence scoring.

ACCENT CATEGORIES:
- American (General American, Southern, New York/Northeast)
- British (RP/Standard, Cockney, Northern England, Scottish)
- Commonwealth (Australian, Canadian, South African)
- International English (Indian, Nigerian, Other)
- Undetermined (insufficient distinctive features)

ANALYSIS REQUIREMENTS:
1. Primary accent classification
2. Confidence score (0-100%)
3. Key acoustic indicators that support your classification
4. Brief explanation of reasoning

Respond in JSON format:
{{
    "accent_type": "Primary classification",
    "accent_subtype": "Specific variant or N/A",
    "confidence_score": 85,
    "english_proficiency_score": 90,
    "key_indicators": ["Specific acoustic evidence"],
    "summary": "Brief explanation of classification reasoning and acoustic evidence"
}}
"""
    
    try:
        model = genaigoogle.GenerativeModel('gemini-2.0-flash')
        # response = model.generate_content(
        #     contents=[prompt,myfile],

        #     generation_config=genaigoogle.types.GenerationConfig(
        #         response_mime_type="application/json",
        #         temperature=0.3
        #     )
        # )

        response = client.models.generate_content(
    model="gemini-2.0-flash", contents=[prompt, myfile]
)
        
        if not response.text:
            return None
        
        # Parse response
        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
        
        return json.loads(result_text)
        # print(response)
        # if not response or not response.text:
        #     st.error("No response from Gemini AI")
        #     return None
        # return response
        
    except Exception as e:
        st.error(f"Gemini analysis error: {e}")
        return None

def cleanup_temp_files(temp_dir):
    """Clean up temporary files"""
    try:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    except Exception as e:
        st.warning(f"Cleanup warning: {e}")

def main():
    # Initialize
    gemini_ok, gemini_msg = init_gemini()
    ffmpeg_ok = check_ffmpeg()
    
    # Status checks
    col1, col2 = st.columns(2)
    with col1:
        if gemini_ok:
            st.success("✅ Gemini AI Ready")
        else:
            st.error(f"❌ Gemini AI: {gemini_msg}")
    
    with col2:
        if ffmpeg_ok:
            st.success("✅ FFmpeg Available")
        else:
            st.warning("⚠️ FFmpeg not found (needed for video URLs)")
    
    if not gemini_ok:
        st.stop()
    
    # Main interface
    st.markdown("---")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["🔗 Video/Audio URL", "📁 Upload Audio File"],
        horizontal=True
    )
    
    if input_method == "🔗 Video/Audio URL":
        url = st.text_input(
            "Enter URL (YouTube, Loom, direct MP4/MP3, etc.):",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        if url and st.button("🎯 Analyze Accent", type="primary"):
            if not ffmpeg_ok and 'youtube' in url.lower():
                st.error("FFmpeg is required for YouTube URLs. Please install FFmpeg or upload an audio file instead.")
                st.stop()
            
            with st.spinner("Downloading and processing audio..."):
                try:
                    # Download audio
                    audio_path, temp_dir = download_audio_from_url(url)
                    st.success("✅ Audio downloaded successfully")
                    
                    # Process audio
                    y, sr = librosa.load(audio_path, sr=16000, duration=120)  # Max 2 minutes
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    if duration < 3:
                        st.warning("Audio is very short. Results may be less accurate.")
                    
                    st.info(f"Processing {duration:.1f} seconds of audio...")
                    
                    # Extract features
                    features = extract_audio_features(y, sr)
                    if not features:
                        st.error("Failed to extract audio features")
                        cleanup_temp_files(temp_dir)
                        st.stop()
                    
                    # Analyze with AI
                    with st.spinner("Analyzing accent with AI..."):
                        result = analyze_with_gemini(features, duration, audio_path)
                        print(result)
                    
                    # Display results
                    if result:
                        st.markdown("---")
                        st.markdown("## 🎯 Accent Analysis Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accent Type", result['accent_type'])
                        with col2:
                            st.metric("Confidence", f"{result['confidence_score']}%")
                        with col3:
                            st.metric("English Proficiency", f"{result.get('english_proficiency_score', 'N/A')}%")
                        
                        if result.get('accent_subtype') and result['accent_subtype'] != 'N/A':
                            st.info(f"**Specific Variant:** {result['accent_subtype']}")
                        
                        st.markdown("### Key Indicators:")
                        for indicator in result.get('key_indicators', []):
                            st.markdown(f"• {indicator}")
                        
                        st.markdown("### Analysis Summary:")
                        st.markdown(result.get('summary', 'No summary available'))
                        
                        # Technical details (expandable)
                        with st.expander("🔧 Technical Details"):
                            st.json({
                                "audio_duration": f"{duration:.1f}s",
                                "sample_rate": f"{sr}Hz",
                                "key_features": {
                                    "mean_pitch": f"{features['mean_pitch']:.1f}Hz",
                                    "tempo": f"{features['tempo']:.1f}BPM",
                                    "formants": [f"{f:.0f}Hz" for f in features['formants']],
                                    "spectral_centroid": f"{features['spectral_centroid_mean']:.1f}Hz"
                                }
                            })
                    else:
                        st.error("Failed to analyze accent. Please try again.")
                    
                    # Cleanup
                    cleanup_temp_files(temp_dir)
                    
                except Exception as e:
                    st.error(f"Error processing audio 1: {str(e)}")
    
    else:  # File upload
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'ogg', 'm4a', 'flac'],
            help="Upload an audio file containing English speech (max 25MB)"
        )
        
        if uploaded_file and st.button("🎯 Analyze Accent", type="primary"):
            with st.spinner("Processing uploaded audio..."):
                try:
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        audio_path = tmp_file.name
                    
                    # Process audio
                    y, sr = librosa.load(audio_path, sr=16000, duration=120)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    if duration < 1:
                        st.error("Audio file is too short for analysis")
                        os.unlink(audio_path)
                        st.stop()
                    
                    st.info(f"Processing {duration:.1f} seconds of audio...")
                    
                    # Extract features
                    features = extract_audio_features(y, sr)
                    # st.write("DEBUG - Features type:", type(features))
                    # st.write("DEBUG - Features content:", features)
                    if not features:
                        st.error("Failed to extract audio features")
                        os.unlink(audio_path)
                        st.stop()
                    
                    # Analyze with AI
                    with st.spinner("Analyzing accent with AI..."):
                        result = analyze_with_gemini(features, duration, audio_path)
                    
                    # Display results (same as above)
                    if result:
                        st.markdown("---")
                        st.markdown("## 🎯 Accent Analysis Results")
                        st.write("DEBUG - Features content:", result)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accent Type", result['accent_type'])
                        with col2:
                            st.metric("Confidence", f"{result['confidence_score']}%")
                        with col3:
                            st.metric("English Proficiency", f"{result['english_proficiency_score']}%")
                        
                        if result.get('accent_subtype') and result['accent_subtype'] != 'N/A':
                            st.info(f"**Specific Variant:** {result['accent_subtype']}")
                        
                        st.markdown("### Key Indicators:")
                        for indicator in result.get('key_indicators', []):
                            st.markdown(f"• {indicator}")
                        
                        st.markdown("### Analysis Summary:")
                        st.markdown(result.get('summary', 'No summary available'))
                        
                        with st.expander("🔧 Technical Details"):
                            formants = features['formants']
                            if hasattr(formants, 'tolist'):  # Check if it's a numpy array
                                formants = formants.tolist()

                            st.json({
                                "audio_duration": f"{duration:.1f}s",
                                "sample_rate": f"{sr}Hz", 
                                "key_features": {
                                    "mean_pitch": f"{features['mean_pitch']:.1f}Hz",
                                    "tempo": f"{features['tempo']:.1f}BPM",
                                    "formants": [f"{f:.0f}Hz" for f in formants],
                                    "spectral_centroid": f"{features['spectral_centroid_mean']:.1f}Hz"
                                }
                            })
                    else:
                        st.error("Failed to analyze accent. Please try again.")
                    
                    # Cleanup
                    os.unlink(audio_path)
                    
                except Exception as e:
                    st.error(f"Error processing audio 2: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, Librosa, and Google Gemini AI*")

if __name__ == "__main__":
    main()