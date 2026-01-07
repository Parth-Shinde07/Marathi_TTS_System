# OCR → Marathi Text → Speech (with ElevenLabs API)

import customtkinter as ctk
from gtts import gTTS
import pygame, os, re, tempfile
import soundfile as sf
import numpy as np
from scipy import signal
from threading import Thread
from tkinter import filedialog, simpledialog
from PIL import Image
import pytesseract
import pdfplumber
import cv2
import time
import requests
import json

pygame.mixer.init()

# -------------------- ElevenLabs Config --------------------
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Available voices - you can get more from the ElevenLabs API
ELEVENLABS_VOICES = {
    "Akshay (Indian Male)": "CZdRaSQ51p0onta4eec8",
    "Riya (Indian Female)": "21m00Tcm4TlvDq8ikWAM",  # Using Rachel as placeholder until custom ID
    "Bill (Indian/Universal)": "pqHfZKP75CvOlQylNhV4",
    "Fin (Universal)": "D38z5RcWu1voky8WS1ja",
    "Rachel (Female)": "21m00Tcm4TlvDq8ikWAM",
    "Adam (Male)": "pNInz6obpgDQGcFmaJgB",
    "Antoni": "ErXwobaYiN019PkySvjV",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Josh": "TxGEqnHWrfWFTfGW9XjX",
    "Sam": "yoZ06aMxZJJ28mfd3POQ",
}

# Config file path
CONFIG_FILE = os.path.expanduser("~/.marathi_tts_config.json")


def load_config():
    """Load saved configuration."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"api_key": "", "voice_id": "21m00Tcm4TlvDq8ikWAM", "use_elevenlabs": False}


def save_config(config):
    """Save configuration."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


class ElevenLabsTTS:
    """ElevenLabs API wrapper for high-quality TTS."""
    
    def __init__(self, api_key, voice_id="21m00Tcm4TlvDq8ikWAM"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = "eleven_multilingual_v2"  # Supports Hindi/Marathi
    
    def set_voice(self, voice_id):
        self.voice_id = voice_id
    
    def set_api_key(self, api_key):
        self.api_key = api_key
    
    def generate(self, text, output_path, stability=0.5, similarity_boost=0.75, style=0.0):
        """Generate speech using ElevenLabs API."""
        if not self.api_key:
            raise ValueError("ElevenLabs API key not set")
        
        url = f"{ELEVENLABS_API_URL}/{self.voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": True
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            error_msg = response.json().get("detail", {}).get("message", "Unknown error")
            raise Exception(f"ElevenLabs API error: {response.status_code} - {error_msg}")
    
    def test_connection(self):
        """Test if API key is valid. Returns (success, message)."""
        # Use /v1/voices endpoint which requires less permissions than /v1/user
        url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                voices_data = response.json()
                voice_count = len(voices_data.get("voices", []))
                return True, f"Connected! {voice_count} voices available"
            elif response.status_code == 401:
                return False, "Invalid API key or missing permissions"
            elif response.status_code == 403:
                return False, "API key expired or forbidden"
            else:
                return False, f"Error: HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except requests.exceptions.ConnectionError:
            return False, "No internet connection"
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"


# -------------------- Prosody (for gTTS fallback) --------------------
class ProsodyModifier:
    """Clean prosody modifier with smooth speed and volume adjustments."""
    
    def apply(self, audio, level, emotion, voice=None, intensity_boost=0.0, sentence_type="statement"):
        """Apply clean prosody modifications without artifacts."""
        speed = 0.95 + (level / 100) * 0.15
        intensity = 0.95 + (level / 100) * 0.15
        volume = 0.90 + (level / 100) * 0.20

        emotion_profiles = {
            "neutral": {"speed": 1.0, "intensity": 1.0, "volume": 1.0},
            "happy": {"speed": 1.05, "intensity": 1.05, "volume": 1.02},
            "sad": {"speed": 0.92, "intensity": 0.90, "volume": 0.95},
            "angry": {"speed": 1.03, "intensity": 1.08, "volume": 1.05},
            "excited": {"speed": 1.06, "intensity": 1.05, "volume": 1.03},
            "calm": {"speed": 0.95, "intensity": 0.95, "volume": 0.95}
        }
        
        profile = emotion_profiles.get(emotion, emotion_profiles["neutral"])
        speed *= profile["speed"]
        intensity *= profile["intensity"]
        volume *= profile["volume"]

        intensity += intensity_boost * 0.3

        if voice:
            speed *= voice["speed"]
            intensity *= voice["intensity"]
            volume *= voice["volume"]

        speed = max(0.85, min(1.20, speed))
        intensity = max(0.80, min(1.20, intensity))
        volume = max(0.70, min(1.15, volume))

        if abs(speed - 1.0) > 0.01:
            new_length = int(len(audio) / speed)
            if new_length > 100:
                audio = signal.resample(audio, new_length)

        mean = np.mean(audio)
        audio = mean + (audio - mean) * intensity
        audio = audio * volume
        
        max_val = np.abs(audio).max()
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        
        return audio


def clean_marathi_ocr(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(
        r"(?:^|\s)([\u0900-\u097F])(?:\s([\u0900-\u097F])){2,}",
        lambda m: m.group(0).replace(" ", ""),
        text
    )
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def generate_silence(duration_sec, sr):
    """Generate clean silence."""
    samples = int(duration_sec * sr)
    return np.zeros(samples)


def get_sentence_type(text):
    """Determine sentence type for intonation patterns."""
    text = text.strip()
    if text.endswith("?") or text.endswith("का") or "का?" in text:
        return "question"
    elif text.endswith("!"):
        return "exclamation"
    else:
        return "statement"


def punctuation_intensity(text):
    """Calculate intensity boost based on punctuation."""
    score = 0
    score += min(text.count("!"), 3) * 0.18
    score += min(text.count("?"), 2) * 0.12
    if "…" in text or "..." in text:
        score += 0.08
    if any(c in text for c in ["॥", "।।"]):
        score += 0.1
    return min(score, 0.5)


def get_pause_duration(text, next_text=None, is_last=False):
    """Get contextual pause duration based on punctuation."""
    text = text.strip()
    
    if is_last:
        return 0.5
    
    if text.endswith(("।", ".", "॥")):
        return 0.4
    elif text.endswith(("!", "?")):
        return 0.45
    elif text.endswith("…") or text.endswith("..."):
        return 0.55
    elif text.endswith(","):
        return 0.2
    elif text.endswith((";", ":")):
        return 0.3
    else:
        return 0.25


# -------------------- Emotion Detection --------------------
EMOTION_KEYWORDS = {
    "happy": [
        "आनंद", "आनंदी", "खुश", "मस्त", "छान", "सुंदर", "अप्रतिम",
        "उत्साह", "हसू", "सुख", "यश", "प्रेम", "आवड", "उत्तम", "धन्यवाद",
        "अभिनंदन", "स्वागत", "प्रसन्न", "भाग्य", "मजा"
    ],
    "sad": [
        "दुःख", "वाईट", "कष्ट", "त्रास", "दुर्दैव", "निराशा", "भीती",
        "ताण", "रड", "अडचण", "निराश", "एकटे", "सोडून", "गेले",
        "मृत्यू", "अपयश", "पाप", "वेदना", "शोक", "उदासीन"
    ],
    "angry": [
        "राग", "चिड", "नको", "थांब", "बास", "मूर्ख", "वेडा",
        "वैताग", "चीड", "भांडण", "उग्र", "संताप", "द्वेष",
        "अपमान", "चूक", "खबरदार", "हिंमत", "बदला"
    ],
    "excited": [
        "वा", "अरे", "बापरे", "किती", "खूप", "जबरदस्त", "भारी",
        "आश्चर्य", "नवल", "कमाल", "धक्का", "वेग"
    ],
    "calm": [
        "शांत", "सावकाश", "हळू", "विश्रांती", "आराम", "समाधान",
        "अर्थ", "विचार", "समजूत", "निसर्ग", "ध्यान"
    ],
    "fear": [
        "भीती", "धोका", "सावधान", "वाचवा", "पळा", "गंभीर"
    ]
}

INTENSIFIERS = ["खूप", "फार", "अतिशय", "अगदी", "नक्कीच", "काय"]


def detect_emotion(text, story=False):
    """Detect emotion with weighted scoring and context analysis."""
    if story:
        return "neutral"

    text_clean = re.sub(r"[^\u0900-\u097F ]", " ", text)
    text_clean = re.sub(r"\s+", " ", text_clean)
    words = text_clean.split()

    scores = {k: 0 for k in EMOTION_KEYWORDS}

    # Keyword scanning
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            # Check for exact word or substring match
            count = text_clean.count(kw)
            if count > 0:
                # Basic score
                score = count * 2.0
                
                # Check for intensifiers before the keyword
                try:
                    idx = text_clean.find(kw)
                    if idx > 5:
                        pre_text = text_clean[max(0, idx-15):idx]
                        if any(inten in pre_text for inten in INTENSIFIERS):
                            score *= 1.5
                except:
                    pass
                
                scores[emotion] += score

    # Punctuation Analysis
    if "!" in text:
        scores["excited"] += 1.5
        scores["angry"] += 1.0
        scores["happy"] += 0.5
    
    if "?" in text:
        scores["excited"] += 0.5
        scores["fear"] += 0.3

    # Length heuristic: very short sentences with ! are often angry or excited
    if len(words) < 4 and "!" in text:
        scores["excited"] += 1.0
        scores["angry"] += 0.5

    best = max(scores, key=scores.get)
    # Threshold to avoid false positives
    return best if scores[best] > 0.5 else "neutral"


# -------------------- Voices --------------------
VOICES = {
    "narration": {"speed": 0.96, "intensity": 0.92, "volume": 0.92},
    "dialogue": {"speed": 1.04, "intensity": 1.02, "volume": 1.0},
    "emphasis": {"speed": 0.9, "intensity": 1.1, "volume": 1.05},
}


def split_into_sentences(text):
    """Split text into sentences with better handling."""
    pattern = r'([^।!?…।।]+[।!?…।।]?)'
    sentences = [s.strip() for s in re.findall(pattern, text) if s.strip()]
    
    merged = []
    buffer = ""
    for s in sentences:
        if len(s) < 5 and buffer:
            buffer += " " + s
        else:
            if buffer:
                merged.append(buffer)
            buffer = s
    if buffer:
        merged.append(buffer)
    
    return merged


# -------------------- OCR --------------------
def extract_text_from_image(path):
    img = Image.open(path)
    text = pytesseract.image_to_string(img, lang="mar")
    return clean_marathi_ocr(text)


def extract_text_from_document(path):
    text = ""
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif path.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""

    return clean_marathi_ocr(text)


def extract_text_from_camera():
    cap = cv2.VideoCapture(0)
    text = ""
    img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("SPACE = capture | ESC = exit", frame)
        k = cv2.waitKey(1)
        if k == 32:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(img, lang="mar")
            break
        elif k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    if img:
        text = pytesseract.image_to_string(img, lang="mar")
    return clean_marathi_ocr(text)


# -------------------- Dialects --------------------
class MarathiDialect:
    def __init__(self, rules):
        self.rules = rules

    def apply(self, text):
        for k, v in self.rules.items():
            text = text.replace(k, v)
        return text


DIALECTS = {
    "Standard": MarathiDialect({}),
    "Varhadi": MarathiDialect({
        "आहे": "आय", "नाही": "नाय", "मी": "म्ही",
        'गा': 'मा', 'ळ': 'ल', 'तू': 'तु',
        'आपण': 'आपुण'
    }),
    "Malvani": MarathiDialect({
        "आहे": "आसा", "नाही": "ना", "मला": "माका",
        'व': 'व्ह', 'तुला': 'तुज्जा', 'पाहिजे': 'पायजे'
    }),
    "Ahirani": MarathiDialect({
        "आहे": "हाय", "नाही": "नाय", 'मला': 'म्हाला',
        'झाला': 'झालं', 'पाहिजे': 'पायजे'
    }),
    "Kokani": MarathiDialect({
        'आहे': 'आसा', 'नाही': 'ना',
        'तुला': 'तुका', 'मला': 'माका', 'पाहिजे': 'जाय'
    })
}


# -------------------- Enhanced TTS --------------------
class MarathiTTS:
    def __init__(self):
        self.prosody = ProsodyModifier()
        self.temp = None
        self.last_emotions = []
        self.timeline = []
        self.sr = None
        
        # Load config
        self.config = load_config()
        self.elevenlabs = ElevenLabsTTS(
            api_key=self.config.get("api_key", ""),
            voice_id=self.config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        )
        self.use_elevenlabs = self.config.get("use_elevenlabs", False)
    
    def set_elevenlabs_config(self, api_key, voice_id, use_elevenlabs):
        """Update ElevenLabs configuration."""
        self.elevenlabs.set_api_key(api_key)
        self.elevenlabs.set_voice(voice_id)
        self.use_elevenlabs = use_elevenlabs
        
        # Save config
        self.config = {
            "api_key": api_key,
            "voice_id": voice_id,
            "use_elevenlabs": use_elevenlabs
        }
        save_config(self.config)

    def generate_with_elevenlabs(self, text, stability=0.5, similarity=0.75, style=0.3):
        """Generate speech using ElevenLabs API."""
        sentences = split_into_sentences(text)
        audios = []
        self.last_emotions.clear()
        self.timeline.clear()
        
        for idx, sent in enumerate(sentences):
            clean_sent = clean_marathi_ocr(sent)
            if not clean_sent:
                continue
            
            emotion = detect_emotion(clean_sent, story=False)
            self.last_emotions.append((clean_sent, emotion))
            
            # Generate audio with ElevenLabs
            mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            
            try:
                # Dynamic style settings based on emotion
                emotion_settings = {
                    "neutral":  {"stability": 0.50, "style": 0.00},
                    "happy":    {"stability": 0.40, "style": 0.60},  # Lower stability = more expressive
                    "sad":      {"stability": 0.85, "style": 0.20},  # High stability = monotone/sad
                    "angry":    {"stability": 0.35, "style": 0.70},  # Very expressive/aggressive
                    "excited":  {"stability": 0.30, "style": 0.80},  # High energy
                    "calm":     {"stability": 0.70, "style": 0.10},  # Stable and soft
                    "fear":     {"stability": 0.35, "style": 0.50}   # Shaky
                }
                
                settings = emotion_settings.get(emotion, emotion_settings["neutral"])
                
                # Blend user slider with preset (slider provides offset)
                # Slider 0-100 maps to -0.2 to +0.2 style boost
                slider_boost = (level - 50) / 100.0 * 0.4
                
                final_stability = max(0.1, min(1.0, settings["stability"] - (slider_boost * 0.5)))
                final_style = max(0.0, min(1.0, settings["style"] + slider_boost))
                
                self.elevenlabs.generate(
                    clean_sent, 
                    mp3_path, 
                    stability=final_stability,
                    similarity_boost=similarity,
                    style=final_style
                )
                
                audio, sr = sf.read(mp3_path)
                self.sr = sr
                
                audios.append(audio)
                
                duration = len(audio) / sr
                self.timeline.append((clean_sent, emotion, duration))
                
                # Add pause
                is_last = (idx == len(sentences) - 1)
                pause_duration = get_pause_duration(clean_sent, None, is_last)
                audios.append(generate_silence(pause_duration, sr))
                
                os.remove(mp3_path)
                
            except Exception as e:
                print(f"ElevenLabs error: {e}")
                os.remove(mp3_path) if os.path.exists(mp3_path) else None
                raise e
        
        if audios:
            final = np.concatenate(audios)
            
            # Normalize
            max_val = np.abs(final).max()
            if max_val > 0:
                final = final / max_val * 0.85
            
            wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            self.temp = wav_path
            sf.write(wav_path, final, sr)

    def generate_with_gtts(self, text, level, dialect, story):
        """Generate speech using gTTS (fallback)."""
        text = DIALECTS[dialect].apply(text)
        sentences = split_into_sentences(text)
        audios = []
        self.last_emotions.clear()
        self.timeline.clear()

        for idx, sent in enumerate(sentences):
            clean_sent = clean_marathi_ocr(sent)
            if not clean_sent:
                continue
                
            emotion = detect_emotion(clean_sent, story)
            sentence_type = get_sentence_type(clean_sent)
            self.last_emotions.append((clean_sent, emotion))

            mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            wav = mp3.replace(".mp3", ".wav")

            try:
                gTTS(clean_sent, lang="mr").save(mp3)
                audio, sr = sf.read(mp3)
                self.sr = sr
            except Exception as e:
                print(f"Error generating audio for: {clean_sent[:30]}... - {e}")
                os.remove(mp3)
                continue
            
            voice = VOICES["dialogue"] if '"' in sent or "'" in sent else VOICES["narration"]

            boost = punctuation_intensity(clean_sent)
            audio = self.prosody.apply(
                audio,
                level,
                emotion,
                voice,
                intensity_boost=boost,
                sentence_type=sentence_type
            )
            
            audios.append(audio)
            
            duration = len(audio) / sr
            self.timeline.append((clean_sent, emotion, duration))

            is_last = (idx == len(sentences) - 1)
            next_sent = sentences[idx + 1] if idx < len(sentences) - 1 else None
            pause_duration = get_pause_duration(clean_sent, next_sent, is_last)
            
            audios.append(generate_silence(pause_duration, sr))

            os.remove(mp3)

        if audios:
            final = np.concatenate(audios)
            
            final = self._apply_compression(final)
            
            max_val = np.abs(final).max()
            if max_val > 0:
                final = final / max_val * 0.85
            
            self.temp = wav
            sf.write(wav, final, sr)

    def generate(self, text, level, dialect, story):
        """Generate speech using configured TTS engine."""
        if self.use_elevenlabs and self.elevenlabs.api_key:
            try:
                # Map level to ElevenLabs parameters
                stability = 0.3 + (level / 100) * 0.4  # Range: 0.3 - 0.7
                similarity = 0.6 + (level / 100) * 0.3  # Range: 0.6 - 0.9
                
                # Apply dialect before generation
                text = DIALECTS[dialect].apply(text)
                
                self.generate_with_elevenlabs(text, stability, similarity)
                return
            except Exception as e:
                print(f"ElevenLabs failed, falling back to gTTS: {e}")
        
        # Fallback to gTTS
        self.generate_with_gtts(text, level, dialect, story)
    
    def _apply_compression(self, audio, threshold=0.6, ratio=3.0):
        """Apply gentle dynamic range compression."""
        output = audio.copy()
        mask = np.abs(audio) > threshold
        output[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
        return output

    def play(self):
        if self.temp and os.path.exists(self.temp):
            pygame.mixer.music.load(self.temp)
            pygame.mixer.music.play()

    def stop(self):
        pygame.mixer.music.stop()


# -------------------- Settings Dialog --------------------
class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, tts):
        super().__init__(parent)
        self.tts = tts
        self.title("⚙️ TTS Settings")
        self.geometry("520x620")
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 520) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 620) // 2
        self.geometry(f"+{x}+{y}")
        
        self._create_widgets()
    
    def _create_widgets(self):
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(frame, text="TTS Engine Settings", font=("Arial", 18, "bold")).pack(pady=10)
        
        # Engine selection
        engine_frame = ctk.CTkFrame(frame, fg_color="transparent")
        engine_frame.pack(fill="x", pady=10)
        
        self.use_elevenlabs = ctk.BooleanVar(value=self.tts.use_elevenlabs)
        ctk.CTkCheckBox(
            engine_frame, 
            text="Use ElevenLabs API (High Quality)", 
            variable=self.use_elevenlabs,
            command=self._toggle_elevenlabs
        ).pack(anchor="w")
        
        ctk.CTkLabel(engine_frame, text="(Unchecked = Use gTTS/Google - Free but lower quality)", 
                     font=("Arial", 10), text_color="gray").pack(anchor="w", padx=25)
        
        # API Key
        self.api_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.api_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(self.api_frame, text="ElevenLabs API Key:").pack(anchor="w")
        self.api_key_entry = ctk.CTkEntry(self.api_frame, width=400, show="*")
        self.api_key_entry.pack(fill="x", pady=5)
        self.api_key_entry.insert(0, self.tts.config.get("api_key", ""))
        
        # Show/Hide button
        self.show_key = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.api_frame, text="Show API Key", variable=self.show_key,
                       command=self._toggle_key_visibility).pack(anchor="w")
        
        # Voice selection
        voice_frame = ctk.CTkFrame(frame, fg_color="transparent")
        voice_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(voice_frame, text="ElevenLabs Voice:").pack(anchor="w")
        
        current_voice_id = self.tts.config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        current_voice_name = "Rachel (Female)"
        for name, vid in ELEVENLABS_VOICES.items():
            if vid == current_voice_id:
                current_voice_name = name
                break
        
        self.voice_var = ctk.StringVar(value=current_voice_name)
        ctk.CTkOptionMenu(voice_frame, values=list(ELEVENLABS_VOICES.keys()), 
                         variable=self.voice_var, width=300).pack(anchor="w", pady=5)
        
        # Test connection button
        self.test_btn = ctk.CTkButton(frame, text="🔗 Test Connection", command=self._test_connection)
        self.test_btn.pack(pady=10)
        
        self.status_label = ctk.CTkLabel(frame, text="", font=("Arial", 11))
        self.status_label.pack(pady=5)
        
        # Get API Key link
        ctk.CTkLabel(frame, text="Get your API key at: elevenlabs.io", 
                    font=("Arial", 10), text_color="#4A90D9").pack(pady=5)
        
        # Save button
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=20)
        
        ctk.CTkButton(btn_frame, text="💾 Save", command=self._save, 
                     fg_color="#28a745", hover_color="#218838", width=120).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="❌ Cancel", command=self.destroy,
                     fg_color="#6c757d", hover_color="#5a6268", width=120).pack(side="right", padx=10)
        
        self._toggle_elevenlabs()
    
    def _toggle_key_visibility(self):
        self.api_key_entry.configure(show="" if self.show_key.get() else "*")
    
    def _toggle_elevenlabs(self):
        state = "normal" if self.use_elevenlabs.get() else "disabled"
        self.api_key_entry.configure(state=state)
        self.test_btn.configure(state=state)
    
    def _test_connection(self):
        api_key = self.api_key_entry.get().strip()
        if not api_key:
            self.status_label.configure(text="❌ Please enter an API key", text_color="red")
            return
        
        self.status_label.configure(text="🔄 Testing connection...", text_color="gray")
        self.update()
        
        test_tts = ElevenLabsTTS(api_key)
        success, message = test_tts.test_connection()
        
        if success:
            self.status_label.configure(text=f"✅ {message}", text_color="green")
        else:
            self.status_label.configure(text=f"❌ {message}", text_color="red")
    
    def _save(self):
        api_key = self.api_key_entry.get().strip()
        voice_name = self.voice_var.get()
        voice_id = ELEVENLABS_VOICES.get(voice_name, "21m00Tcm4TlvDq8ikWAM")
        use_elevenlabs = self.use_elevenlabs.get()
        
        self.tts.set_elevenlabs_config(api_key, voice_id, use_elevenlabs)
        
        self.destroy()


# -------------------- GUI --------------------
class OCRTTSApp:
    def __init__(self):
        self.tts = MarathiTTS()
        self.text = ""

        self.win = ctk.CTk()
        self.win.geometry("750x650")
        self.win.title("मराठी OCR TTS - ElevenLabs Powered")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        frame = ctk.CTkFrame(self.win)
        frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title with settings button
        title_frame = ctk.CTkFrame(frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="मराठी Text-to-Speech", font=("Arial", 20, "bold")).pack(side="left")
        ctk.CTkButton(title_frame, text="⚙️ Settings", width=100, command=self._open_settings,
                     fg_color="#6c757d", hover_color="#5a6268").pack(side="right")
        
        # Engine indicator
        self.engine_label = ctk.CTkLabel(frame, text="", font=("Arial", 11))
        self.engine_label.pack(pady=2)
        self._update_engine_label()

        # Input buttons frame
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=5)
        
        ctk.CTkButton(btn_frame, text="📁 Image / Document", command=self.select_input, width=200).pack(side="left", padx=5, expand=True)
        ctk.CTkButton(btn_frame, text="📷 Camera OCR", command=self.camera_input, width=200).pack(side="right", padx=5, expand=True)

        # Settings frame
        settings_frame = ctk.CTkFrame(frame, fg_color="transparent")
        settings_frame.pack(fill="x", pady=10)
        
        left_settings = ctk.CTkFrame(settings_frame, fg_color="transparent")
        left_settings.pack(side="left", expand=True)
        
        self.story = ctk.BooleanVar()
        ctk.CTkCheckBox(left_settings, text="📖 Story Mode (neutral emotion)", variable=self.story).pack()

        right_settings = ctk.CTkFrame(settings_frame, fg_color="transparent")
        right_settings.pack(side="right", expand=True)
        
        ctk.CTkLabel(right_settings, text="Dialect:").pack(side="left", padx=5)
        self.dialect = ctk.StringVar(value="Standard")
        ctk.CTkOptionMenu(right_settings, values=list(DIALECTS.keys()), variable=self.dialect, width=120).pack(side="left")

        # Expressiveness slider
        slider_frame = ctk.CTkFrame(frame, fg_color="transparent")
        slider_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(slider_frame, text="🎭 Expressiveness:").pack(side="left", padx=5)
        self.slider = ctk.CTkSlider(slider_frame, from_=0, to=100, width=300)
        self.slider.set(60)
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        self.slider_value = ctk.CTkLabel(slider_frame, text="60%", width=50)
        self.slider_value.pack(side="right")
        self.slider.configure(command=self._update_slider_label)

        # Control buttons
        ctrl_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ctrl_frame.pack(fill="x", pady=10)
        
        ctk.CTkButton(ctrl_frame, text="▶ Speak", command=self.speak, fg_color="#28a745", hover_color="#218838", width=150).pack(side="left", padx=10, expand=True)
        ctk.CTkButton(ctrl_frame, text="⏹ Stop", command=self.tts.stop, fg_color="#dc3545", hover_color="#c82333", width=150).pack(side="right", padx=10, expand=True)

        # Emotion display
        self.emotion_label = ctk.CTkLabel(frame, text="🎭 Detected Emotion: -", font=("Arial", 12))
        self.emotion_label.pack(pady=5)

        # Text display box
        self.box = ctk.CTkTextbox(frame, height=200, font=("Arial", 14))
        self.box.pack(fill="both", expand=True, pady=10)
        self.box.tag_config("highlight", background="#FFD966", foreground="#000000")
        self.box.configure(state="disabled")
        
        # Status bar
        self.status = ctk.CTkLabel(frame, text="Ready", font=("Arial", 10), text_color="gray")
        self.status.pack(pady=5)

    def _open_settings(self):
        SettingsDialog(self.win, self.tts)
        self._update_engine_label()
    
    def _update_engine_label(self):
        if self.tts.use_elevenlabs and self.tts.elevenlabs.api_key:
            self.engine_label.configure(text="🔊 Engine: ElevenLabs (High Quality)", text_color="#28a745")
        else:
            self.engine_label.configure(text="🔊 Engine: Google TTS (Free)", text_color="#ffc107")

    def _update_slider_label(self, value):
        self.slider_value.configure(text=f"{int(value)}%")

    def highlight_sentence(self, sentence, emotion):
        self.box.configure(state="normal")
        self.box.tag_remove("highlight", "1.0", "end")
        idx = self.box.search(sentence, "1.0", stopindex="end")
        if idx:
            end = f"{idx}+{len(sentence)}c"
            self.box.tag_add("highlight", idx, end)
            self.box.see(idx)
        self.box.configure(state="disabled")
        
        emotion_emojis = {
            "happy": "😊", "sad": "😢", "angry": "😠", 
            "excited": "🤩", "calm": "😌", "neutral": "😐"
        }
        emoji = emotion_emojis.get(emotion, "😐")
        self.emotion_label.configure(text=f"🎭 Detected Emotion: {emoji} {emotion.capitalize()}")

    def update_ui(self, text):
        self.text = text
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")

        self.sentences = split_into_sentences(text)

        for s in self.sentences:
            self.box.insert("end", s + " ")

        self.box.configure(state="disabled")
        self.status.configure(text=f"Loaded {len(self.sentences)} sentences")

    def select_input(self):
        path = filedialog.askopenfilename(
            parent=self.win,
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("Docs", "*.pdf *.txt")]
        )
        if not path:
            return

        self.status.configure(text="Processing...")

        def task():
            text = extract_text_from_image(path) if path.endswith((".png", ".jpg", ".jpeg")) else extract_text_from_document(path)
            self.win.after(0, lambda: self.update_ui(text))

        Thread(target=task, daemon=True).start()

    def camera_input(self):
        self.status.configure(text="Opening camera...")
        Thread(
            target=lambda: self.win.after(0, lambda: self.update_ui(extract_text_from_camera())),
            daemon=True
        ).start()

    def speak(self):
        if not self.text:
            self.status.configure(text="No text to speak!")
            return

        level = self.slider.get()
        dialect = self.dialect.get()
        story = self.story.get()
        
        engine = "ElevenLabs" if self.tts.use_elevenlabs and self.tts.elevenlabs.api_key else "Google TTS"
        self.status.configure(text=f"Generating speech with {engine}...")
        self._update_engine_label()

        def task():
            try:
                self.tts.generate(self.text, level, dialect, story)
                self.win.after(0, lambda: self.status.configure(text="Speaking..."))
                self.tts.play()

                for sent, emo, duration in self.tts.timeline:
                    self.win.after(0, lambda s=sent, e=emo: self.highlight_sentence(s, e))
                    time.sleep(duration + 0.3)
                
                self.win.after(0, lambda: self.status.configure(text="Done"))
                self.win.after(0, lambda: self.emotion_label.configure(text="🎭 Detected Emotion: -"))
            except Exception as e:
                self.win.after(0, lambda: self.status.configure(text=f"Error: {str(e)[:50]}"))

        Thread(target=task, daemon=True).start()

    def run(self):
        self.win.mainloop()


if __name__ == "__main__":
    OCRTTSApp().run()
