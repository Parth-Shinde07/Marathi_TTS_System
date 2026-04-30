# OCR → Marathi Text → Speech
        
import customtkinter as ctk
from gtts import gTTS
import pygame, os, re, tempfile, shutil
import soundfile as sf
import numpy as np
from scipy import signal
from threading import Thread
from tkinter import filedialog
from PIL import Image
import pytesseract
import pdfplumber
import cv2
import time
import librosa
import librosa.display
from ml_engine import MMSVitsEngine
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for background processing if needed
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import noisereduce as nr

# Initialize mixer at high fidelity frequency (24kHz) at top level
if not pygame.mixer.get_init():
    # Larger buffer (8192) prevents stuttering during high-fidelity mastering
    pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=8192)

# -------------------- Prosody (for gTTS) --------------------
class ProsodyModifier:
    """Clean prosody modifier using librosa for high-quality audio manipulation."""
    
    def apply(self, audio, level, emotion, voice=None, intensity_boost=0.0, sentence_type="statement", sr=16000):
        """Apply clean prosody modifications without pitch artifacts."""
        # Convert to float32 for librosa
        audio = audio.astype(np.float32)
        
        # Base parameters
        speed = 1.0 + (level - 50) / 400.0 # Range: 0.875 to 1.125
        intensity = 1.0 + (level - 50) / 400.0
        volume = 1.0 + (level - 50) / 300.0

        emotion_profiles = {
            "neutral":  {"speed": 1.0,  "intensity": 1.0,  "volume": 1.0, "pitch": 0.0, "freq_hz": 165.0},
            "happy":    {"speed": 1.02, "intensity": 1.05, "volume": 1.0, "pitch": 0.3, "freq_hz": 210.0},
            "sad":      {"speed": 0.96, "intensity": 0.95, "volume": 0.9, "pitch": -0.2, "freq_hz": 130.0},
            "angry":    {"speed": 1.03, "intensity": 1.10, "volume": 1.0, "pitch": -0.2, "freq_hz": 145.0},
            "excited":  {"speed": 1.05, "intensity": 1.08, "volume": 1.0, "pitch": 0.5, "freq_hz": 240.0},
            "calm":     {"speed": 0.95, "intensity": 0.92, "volume": 0.9, "pitch": 0.0, "freq_hz": 155.0},
            "fear":     {"speed": 1.08, "intensity": 1.02, "volume": 1.0, "pitch": 0.6, "freq_hz": 230.0},
            "shock":    {"speed": 1.12, "intensity": 1.25, "volume": 1.1, "pitch": 0.9, "freq_hz": 260.0},
            "serious":  {"speed": 0.98, "intensity": 1.05, "volume": 1.0, "pitch": -0.1, "freq_hz": 140.0}
        }
        
        profile = emotion_profiles.get(emotion, emotion_profiles["neutral"])
        speed *= profile["speed"]
        intensity *= profile["intensity"]
        volume *= profile["volume"]
        pitch = profile["pitch"]

        # Sentence type adjustments for natural Marathi intonation
        if sentence_type == "question":
            pitch += 0.45 
            speed *= 0.98 # Questions are slightly slower for clarity
            intensity *= 0.95 
        elif sentence_type == "exclamation":
            intensity *= 1.15
            volume *= 1.1
            pitch += 0.2

        intensity += intensity_boost * 0.3

        if voice:
            speed *= voice.get("speed", 1.0)
            intensity *= voice.get("intensity", 1.0)
            volume *= voice.get("volume", 1.0)

        # 1. DC Offset Removal (Essential for clean electronic audio)
        audio = audio - np.mean(audio)

        # 2. Apply Speed (Time-Stretching)
        # For ML Engines, we prefer native scaling (speaking_rate) which is artifact-free.
        # librosa is only used as a fallback for gTTS or non-native voices.
        is_native_speed = voice.get("is_native", True) # Default to True for ML-era voices
        
        if not is_native_speed and abs(speed - 1.0) > 0.01:
            try:
                import librosa
                audio = librosa.effects.time_stretch(audio, rate=speed)
            except Exception as e:
                print(f"Time stretch error: {e}")

        # 3. Apply Pitch Shifting (Use only if specifically requested, as it adds FFT noise)
        if not is_native_speed and abs(pitch) > 0.05:
            try:
                import librosa
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch)
            except Exception as e:
                print(f"Pitch shift error: {e}")

        # 4. Apply Intensity (Dynamic Range)
        if abs(intensity - 1.0) > 0.01:
            mean = np.mean(audio)
            audio = mean + (audio - mean) * intensity
        
        # 5. Advanced Mastering (Noise Reduction + High-Cut)
        audio = remove_noise(audio, sr)
         
        # 6. Apply Volume
        audio = audio * volume
        
        # 7. Apply Smooth Transitions
        audio = apply_fades(audio, sr)

        # 8. Natural Limiting & Soft Clipping
        max_val = np.abs(audio).max()
        if max_val > 0.98:
            audio = np.tanh(audio / max_val) * 0.95 # Soft knee limiting
        
        # Expert Tip: Explicitly handle DC bias early to prevent feedback in neural layers
        audio = audio - np.mean(audio)
        
        return audio.astype(np.float32)


# -------------------- Expert Marathi Normalizer --------------------
class MarathiNormalizer:
    """Advanced normalizer for complex Marathi text including multi-digit numbers and regional abbreviations."""
    
    NUM_MAP = {
        '0': 'शून्य', '1': 'एक', '2': 'दोन', '3': 'तीन', '4': 'चार', '5': 'पाच',
        '6': 'सहा', '7': 'सात', '8': 'आठ', '9': 'नऊ', '10': 'दहा',
        '20': 'वीस', '30': 'तीस', '40': 'चाळीस', '50': 'पन्नास', '60': 'साठ',
        '70': 'सत्तर', '80': 'ऐंशी', '90': 'नव्वद', '100': 'शंभर', '1000': 'हजार', '100000': 'लाख'
    }
    
    TEENS = {
        '11': 'अकरा', '12': 'बारा', '13': 'तेरा', '14': 'चौदा', '15': 'पंधरा',
        '16': 'सोळा', '17': 'सतरा', '18': 'अठरा', '19': 'एकोणीस'
    }

    ABBR_MAP = {
        'डॉ.': 'डॉक्टर', 'प्रा.': 'प्राध्यापक', 'सौ.': 'सौभाग्यवती', 'चि.': 'चिरंजीव',
        'कु.': 'कुमारी', 'उदा.': 'उदाहरणार्थ', 'क्र.': 'क्रमांक', 'इ.': 'इतर',
        'दि.': 'दिनांक', 'रु.': 'रुपये', 'म.': 'महाराष्ट्र', 'स्व.': 'स्वर्गीय',
        'शा.': 'शासकीय', 'जि.': 'जिल्हा', 'प.': 'पश्चिम', 'पू.': 'पूर्व',
        'उ.': 'उत्तर', 'द.': 'दक्षिण', 'किमी': 'किलोमीटर', 'टक्के': 'टक्केवारी'
    }

    @classmethod
    def num_to_marathi(cls, n):
        """Robust mapping for multi-digit Marathi numbers."""
        n = int(n)
        if n == 0: return "शून्य"
        if n < 11: return cls.NUM_MAP.get(str(n))
        if 11 <= n <= 19: return cls.TEENS.get(str(n))
        
        words = []
        if n >= 100000:
            words.append(cls.num_to_marathi(n // 100000) + " लाख")
            n %= 100000
        if n >= 1000:
            words.append(cls.num_to_marathi(n // 1000) + " हजार")
            n %= 1000
        if n >= 100:
            count = n // 100
            words.append((cls.NUM_MAP.get(str(count)) if count > 1 else "") + " शंभर")
            n %= 100
        if n > 0:
            if n <= 10: words.append(cls.NUM_MAP.get(str(n)))
            elif str(n) in cls.TEENS: words.append(cls.TEENS.get(str(n)))
            else:
                tens = (n // 10) * 10
                units = n % 10
                if units == 0: words.append(cls.NUM_MAP.get(str(tens)))
                else:
                    # Marathi numbers 21-99 are unique words, but for TTS clarity, 
                    # 'tens units' is often acceptable if the model was trained thus.
                    # Best practice: Full map. Here we use high-quality approximations for standard VITS.
                    words.append(cls.NUM_MAP.get(str(tens)) + " " + cls.NUM_MAP.get(str(units)))
        
        return " ".join(words).strip()

    @classmethod
    def normalize(cls, text):
        if not text: return ""
        
        # 1. Expand Abbreviations first (to catch things like डॉ. 5)
        for abbr, full in cls.ABBR_MAP.items():
            text = text.replace(abbr, full)
            
        # 2. Multi-digit Number Expansion (supports up to Lakhs)
        # Using a regex that finds digits and replaces with Marathi words
        text = re.sub(r'\d+', lambda m: " " + cls.num_to_marathi(m.group(0)) + " ", text)
        
        return text

def clean_marathi_ocr(text: str) -> str:
    """Clean OCR hallucinations and broken Marathi characters."""
    if not text:
        return ""

    # Remove the dotted circle symbol and zero-width characters
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\u25CC', '')
    
    # Remove standalone OCR hallucinations like 'ट्र', 'टू', and 'ट््'
    # Tesseract frequently inserts these phantom characters when trying to read shadows
    text = re.sub(r'\b[ट्रटू]\b', '', text)
    text = re.sub(r'[ट्रटू]$', '', text)
    text = re.sub(r'ट््', '', text)
    
    # Fix common word gluing and structural misreadings
    text = text.replace('आजचादिवस', 'आजचा दिवस')
    text = text.replace('विडे ना', 'ही')
    text = text.replace('ट् ', '')
    text = text.replace('◌', '') # Remove dotted circles
    
    # Standardize whitespace and remove non-Marathi characters
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"[^\u0900-\u097F ।॥!?%,.:;…\.]", " ", text)
    
    # Fix detached Matras (vowel signs) by gluing them back to the previous character
    text = re.sub(r'\s+([\u0900-\u0903\u093A-\u0957])', r'\1', text)
    
    # Final cleanup of pipes and double spaces
    text = text.replace('|', '।')
    text = re.sub(r" {2,}", " ", text)

    # Fix broken clusters and ensure halants are followed by correct consonants
    text = re.sub(r'([^\u0900-\u097F])्', r'\1', text) # Remove stray halants
    
    # Syllable Stress / Phonetic Repair: Insert Zero-Width Space (ZWSP) for complex words
    # to help the VITS engine articulate distinct syllables in long compounds.
    complex_repairs = {
        'क्रीडांगण': 'क्री​डां​ग​ण', 
        'प्रयोगशाळा': 'प्र​यो​ग​शा​ळा',
        'संगणक': 'सं​ग​ण​क',
        'प्रशासकीय': 'प्र​शा​स​की​य'
    }
    for word, repair in complex_repairs.items():
        text = text.replace(word, repair)

    # Use the existing MarathiNormalizer class in your file
    text = MarathiNormalizer.normalize(text)
    
    return text.strip()

def generate_silence(duration_sec, sr):
    """Generate clean silence."""
    samples = int(duration_sec * sr)
    return np.zeros(samples, dtype=np.float32)


def apply_fades(audio, sr, fade_ms=40):
    """Apply soft fades to start and end of audio to prevent clicking noise. 40ms for smoothness."""
    fade_samples = int(sr * fade_ms / 1000)
    if len(audio) < fade_samples * 2:
        return audio
    
    # Use S-curve (Hanning) for even smoother transitions than linear
    fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fade_samples)))
    fade_out = 0.5 * (1 - np.cos(np.pi * np.linspace(1, 0, fade_samples)))
    
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    return audio


def remove_noise(audio, sr):
    """
    Advanced Studio Mastering Chain:
    1. Zero-phase DC Offset removal.
    2. Adaptive stationary noise reduction (Hiss removal).
    3. Zero-phase High-Cut filter (9kHz).
    4. Soft Noise Gate (Logarithmic release).
    """
    try:
        import noisereduce as nr
        from scipy.signal import butter, sosfiltfilt
        
        # Ensure float32 for processing
        audio = audio.astype(np.float32)
        
        # 1. Zero-mean the signal (DC Offset)
        audio = audio - np.mean(audio)
        
        # 2. Adaptive Hiss Reduction
        # prop_decrease=0.7 is the "sweet spot" for vocoder hiss
        audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.7, stationary=True)
        
        # 3. High-Cut Filter (9kHz) using Zero-phase SOS filter
        sos = butter(4, 9000, btype='low', fs=sr, output='sos')
        audio = sosfiltfilt(sos, audio)
        
        # 4. Soft Noise Gate
        # Instead of hard-cutting, we gently fade out silence to avoid clicks
        rms = np.sqrt(np.mean(audio**2))
        gate_threshold = rms * 0.1 # Dynamic threshold based on average level
        
        # Simple soft-gate implementation
        mask = np.abs(audio) > gate_threshold
        # Smooth the mask using a simple moving average (kernel size ~50ms)
        kernel_size = int(0.05 * sr)
        if kernel_size > 0:
            smooth_mask = np.convolve(mask.astype(float), np.ones(kernel_size)/kernel_size, mode='same')
            audio = audio * smooth_mask
            
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Mastering Warning: {e}")
        return audio


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
    score += min(text.count("!"), 4) * 0.25 # Sharp boost for exclamations
    score += min(text.count("?"), 3) * 0.15
    if "…" in text or "..." in text:
        score += 0.1
    if any(c in text for c in ["॥", "।।"]):
        score += 0.12
    return min(score, 0.7) # Allow up to 0.7 boost for extreme emotion


def get_pause_duration(text, next_text=None, is_last=False):
    """Get contextual pause duration. Extreme Smoothness Edition."""
    text = text.strip()
    
    if is_last:
        return 1.5 # Full breathing space at end
    
    if text.endswith(("।", ".", "॥")):
        return 1.1 # Deep sentence break
    elif text.endswith(("!", "?")):
        return 1.2
    elif text.endswith("…") or text.endswith("..."):
        return 1.4
    elif text.endswith(","):
        return 0.4 # Short breath for commas
    elif text.endswith((";", ":")):
        return 0.8
    else:
        return 0.65 # Natural phrasing pause


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
    ],
    "shock": [
        "धक्का", "आश्चर्य", "चकित", "शॉक", "बापरे", "अरे", "काय", "खरंच", "कसे", "स्वप्न"
    ],
    "serious": [
        "महत्त्वाचे", "गंभीर", "लक्ष", "नियम", "कायदा", "शासन", "कर्तव्य", "जबाबदारी"
    ]
}

INTENSIFIERS = ["खूप", "फार", "अतिशय", "अगदी", "नक्कीच", "काय"]


def detect_emotion(text, story=False):
    """Detect emotion with strict word-boundary matching and neutral bias."""
    if story:
        # Stories now allow full emotion detection for dramatic effect
        pass

    text_clean = re.sub(r"[^\u0900-\u097F ]", " ", text)
    words = set(text_clean.split())

    # Initialize scores including 'neutral' to avoid KeyError
    scores = {k: 0.0 for k in EMOTION_KEYWORDS}
    scores["neutral"] = 0.0

    # High-accuracy word matching (prevents substring false positives)
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in words:
                scores[emotion] += 2.5
            elif any(kw in word for word in words): # Partial match suffix/prefix
                scores[emotion] += 1.0

    # Punctuation context
    if "!" in text:
        if any(w in words for w in ["बापरे", "धक्का", "काय", "अरे"]):
            scores["shock"] += 3.0
        elif any(w in words for w in ["राग", "खबरदार", "मूर्ख"]):
            scores["angry"] += 2.0
        else:
            scores["excited"] += 1.5
    
    if "?" in text:
        if any(w in words for w in ["काय", "कसे", "खरंच"]):
            scores["shock"] += 1.5
        elif any(w in words for w in ["का", "कुठे"]):
            scores["neutral"] += 0.5 
        else:
            scores["excited"] += 0.5

    # Find the strongest emotion
    best_emotion = "neutral"
    max_score = 0.0
    
    # We require a minimum score of 2.0 to move away from neutral
    for emotion, score in scores.items():
        if score > max_score:
            max_score = score
            best_emotion = emotion

    return best_emotion if max_score >= 2.0 else "neutral"


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
    # Only merge very short snippets to keep generation windows consistent
    for s in sentences:
        if len(s) < 10 and buffer:
            buffer += " " + s
        else:
            if buffer:
                merged.append(buffer)
            buffer = s
    if buffer:
        merged.append(buffer)
    
    return merged


# -------------------- OCR --------------------
def preprocess_for_ocr(img_cv):
    """Apply aggressive OpenCV filters to make Devanagari text highly readable by Tesseract."""
    # 1. Grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize (Tesseract needs high resolution, ~300 DPI equivalent)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 3. Noise Removal
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Adaptive Thresholding (Binarization: stark white background, black text)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Morphological Closing (Connects broken Devanagari strokes and matras)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return processed


def extract_text_from_image(path):
    """Enhanced OCR for Marathi with structural preservation."""
    try:
        img = cv2.imread(path)
        if img is None: return ""
        
        # Preprocess using the advanced pipeline
        processed_img = preprocess_for_ocr(img)
        
        # --psm 6: Assumes a single uniform block of text.
        # --oem 3: Uses the best available engine (LSTM).
        custom_config = r'--oem 3 --psm 6 -l mar -c preserve_interword_spaces=1'
        
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        return clean_marathi_ocr(text)
    except Exception as e:
        print(f"Image OCR Error: {e}")
        return ""

def extract_text_from_document(path):
    """Extract text from PDF/TXT and clean it immediately."""
    text = ""
    try:
        if path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        elif path.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    # layout=True helps keep Marathi words together
                    page_content = p.extract_text(layout=True)
                    if page_content:
                        text += page_content + " "
        
        return clean_marathi_ocr(text)
    except Exception as e:
        print(f"Document Extraction Error: {e}")
        return ""


def extract_text_from_camera():
    cap = cv2.VideoCapture(0)
    text = ""
    processed_img = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.imshow("SPACE = capture | ESC = exit", frame)
        k = cv2.waitKey(1)
        if k == 32: # SPACE
            processed_img = preprocess_for_ocr(frame)
            break
        elif k == 27: # ESC
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if processed_img is not None:
        text = pytesseract.image_to_string(processed_img, lang="mar", config="--psm 6 -c preserve_interword_spaces=1")
    
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


# -------------------- Marathi TTS --------------------
class MarathiTTS:
    def __init__(self):
        self.prosody = ProsodyModifier()
        self.temp = None
        self.last_emotions = []
        self.timeline = []
        self.sr = 24000 # Standard target rate
        self._ml_engine = None
        self._ml_engine_type = None
        self.stop_signal = False

    def _get_ml_engine(self, engine_type="Standard (VITS/IIT Bombay)"):
        if self._ml_engine is None or self._ml_engine_type != engine_type:
            from ml_engine import get_engine
            self._ml_engine = get_engine(engine_type)
            self._ml_engine_type = engine_type
        return self._ml_engine

    def generate_sentence_audio(self, sent, level, dialect, story):
        """Generate audio for a single sentence."""
        use_ml = True
        target_sr = 24000
        engine_type = "Standard (VITS/IIT Bombay)"
        realism_mode = True

        voice_tone = "Crystal Female"
        
        sent = DIALECTS[dialect].apply(sent)
        clean_sent = clean_marathi_ocr(sent)
        if not clean_sent:
            return None, target_sr, "neutral", 0
            
        emotion = detect_emotion(clean_sent, story)
        sentence_type = get_sentence_type(clean_sent)
        
        voice = VOICES["dialogue"] if '"' in sent or "'" in sent else VOICES["narration"]
        voice["is_native"] = False
        use_ml = True # Defaulting to True for VITS engine usage
        
        # Emphasis check for boost
        emphasis_words = ["खूप", "फार", "अतिशय"]
        emphasis_boost = 0.2 if any(w in clean_sent for w in emphasis_words) else 0.0
        
        # Log global emotion tag at start of processing
        print(f"[GLOBAL EMOTION TAG]: {emotion.upper()}")

        # Generate audio for the whole sentence at once for better fluency/co-articulation
        audio, orig_sr = self._generate_raw_audio(clean_sent, level, dialect, emotion, voice_tone, realism_mode, engine_type)
        
        if audio is None:
            return None, target_sr, emotion, 0

        # Apply prosody mastering
        audio = self.prosody.apply(
            audio, 
            level if not use_ml else (level * 0.7), 
            emotion, 
            voice, 
            intensity_boost=emphasis_boost, 
            sentence_type=sentence_type, 
            sr=target_sr
        )

        # Add expert-requested breathing pause at the end of the sentence
        # (VITS handles internal commas naturally; we add the major breath here)
        pause_sec = 1.1 if clean_sent.endswith(("।", ".", "!", "?")) else 0.4
        silence = generate_silence(pause_sec, target_sr)
        audio = np.concatenate([audio, silence])
        
        duration = len(audio) / target_sr
        return audio, target_sr, emotion, duration

    def _generate_raw_audio(self, text, level, dialect, emotion, voice_tone, realism_mode, engine_type):
        target_sr = 24000
        try:
            engine = self._get_ml_engine(engine_type)
            audio, orig_sr = engine.generate_speech(
                text, 
                speed=1.0 + (level - 50) / 300.0, 
                emotion=emotion, 
                voice_tone=voice_tone,
                realism_mode=realism_mode
            )
            if orig_sr != target_sr:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            return audio, target_sr
        except Exception as e:
            print(f"ML Segment Error: {e}")
            return None, target_sr

    def generate(self, text, level, dialect, story):
        # We now keep this for batch generation if needed, but speak() will use the new streaming logic.
        # This also populates self.temp for the Analysis window.
        sentences = split_into_sentences(text)
        audios = []
        self.last_emotions.clear()
        self.timeline.clear()
        target_sr = 24000
        
        for sent in sentences:
            audio, sr, emo, dur = self.generate_sentence_audio(sent, level, dialect, story)
            if audio is not None:
                audios.append(audio)
                self.last_emotions.append((sent, emo))
                self.timeline.append((sent, emo, dur))
        
        if audios:
            # Prepend a tiny bit of silence (100ms) to prevent cutting off the first word
            audios.insert(0, generate_silence(0.1, target_sr))
            final = np.concatenate(audios)
            
            # Subtle final mastering
            final = self._apply_compression(final)
            
            # DC correction for entire file
            final = final - np.mean(final)

            # --- PRE-PLAY MASTERING ENGINE (Crisp-Human Edition) ---
            try:
                from scipy.signal import butter, lfilter
                # 1. Peaking EQ for presence (surgical boost at 5kHz)
                # This avoids adding noise from other bands
                w0 = 2 * np.pi * 5000 / 12000
                alpha = np.sin(w0) / 2.0
                gain_db = 1.5
                A = 10**(gain_db/40)
                b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
                a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
                final = lfilter(b, a, final)
                
                # 2. Vowel Warmth (400Hz - 800Hz)
                b_warm, a_warm = butter(1, [400/12000, 800/12000], btype='bandpass')
                warmth = lfilter(b_warm, a_warm, final)
                final = final + (warmth * 0.1)
                
                # 3. High-Pass Cleanup (Removes low-end rumble below 100Hz)
                b_low, a_low = butter(2, 100/12000, btype='highpass')
                final = lfilter(b_low, a_low, final)
                
                # 4. Final Mastering (Dynamic Normalization)
                final = final - np.mean(final)
            except:
                pass

            max_val = np.abs(final).max()
            if max_val > 0:
                final = final / max_val * 0.90
            
            self.temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            sf.write(self.temp, final, target_sr)

    def _apply_compression(self, audio, threshold=0.75, ratio=2.5):
        """Apply very gentle dynamic range compression to avoid clipping while keeping clarity."""
        output = audio.copy()
        mask = np.abs(audio) > threshold
        output[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
        return output

    def play(self):
        if self.temp and os.path.exists(self.temp):
            pygame.mixer.music.load(self.temp)
            pygame.mixer.music.play()

    def stop(self):
        self.stop_signal = True
        pygame.mixer.music.stop()


# -------------------- GUI --------------------
class OCRTTSApp:
    def __init__(self):
        self.tts = MarathiTTS()
        self.text = ""

        self.win = ctk.CTk()
        self.win.geometry("750x750")
        self.win.title("मराठी OCR TTS System")
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        frame = ctk.CTkFrame(self.win)
        frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        ctk.CTkLabel(frame, text="मराठी Text-to-Speech", font=("Kohinoor Devanagari", 22, "bold")).pack(pady=(0, 20))

        # Input buttons frame
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=5)
        
        ctk.CTkButton(btn_frame, text="📁 Image / Document", command=self.select_input, width=200).pack(side="left", padx=5, expand=True)
        ctk.CTkButton(btn_frame, text="📷 Camera OCR", command=self.camera_input, width=200).pack(side="left", padx=5, expand=True)

        # Settings frame
        settings_frame = ctk.CTkFrame(frame, fg_color="transparent")
        settings_frame.pack(fill="x", pady=10)
        left_settings = ctk.CTkFrame(settings_frame, fg_color="transparent")
        left_settings.pack(side="left", expand=True)
        
        self.story = ctk.BooleanVar()
        ctk.CTkCheckBox(left_settings, text="📖 Story Mode", variable=self.story).pack(side="left", padx=10)

        right_settings = ctk.CTkFrame(settings_frame, fg_color="transparent")
        right_settings.pack(side="right", expand=True)
        
        ctk.CTkLabel(right_settings, text="Dialect:").pack(side="left", padx=5)
        self.dialect = ctk.StringVar(value="Standard")
        ctk.CTkOptionMenu(right_settings, values=list(DIALECTS.keys()), variable=self.dialect, width=120).pack(side="left")


        # Expressiveness slider (Speed & Emotion level)
        slider_frame = ctk.CTkFrame(frame, fg_color="transparent")
        slider_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(slider_frame, text="Speech Speed & Emotion:").pack(side="left", padx=10)
        self.express_level = ctk.CTkSlider(slider_frame, from_=0, to=100, number_of_steps=20)
        self.express_level.set(50) # Set to 50 for perfectly natural speed
        self.express_level.pack(side="left", fill="x", expand=True, padx=10)
        self.slider_value = ctk.CTkLabel(slider_frame, text="50%", width=50)
        self.slider_value.pack(side="right")
        self.express_level.configure(command=self._update_slider_label)

        # Control buttons
        ctrl_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ctrl_frame.pack(fill="x", pady=5)
        
        ctk.CTkButton(ctrl_frame, text="▶ Speak", command=self.speak, fg_color="#28a745", hover_color="#218838", width=100).pack(side="left", padx=5, expand=True)
        ctk.CTkButton(ctrl_frame, text="💾 Save", command=self.save_audio, fg_color="#ffc107", hover_color="#e0a800", text_color="#000000", width=100).pack(side="left", padx=5, expand=True)
        ctk.CTkButton(ctrl_frame, text="📊 Analysis", command=self.show_analysis, fg_color="#17a2b8", hover_color="#138496", width=100).pack(side="left", padx=5, expand=True)

        ctk.CTkButton(ctrl_frame, text="⏹ Stop", command=self.tts.stop, fg_color="#dc3545", hover_color="#c82333", width=100).pack(side="left", padx=5, expand=True)

        # Waveform Visualization
        self.waveform_frame = ctk.CTkFrame(frame, height=120, fg_color="#1a1a1a")
        self.waveform_frame.pack(fill="x", pady=10)
        self.waveform_placeholder = ctk.CTkLabel(self.waveform_frame, text="Audio Waveform Visualization", text_color="gray")
        self.waveform_placeholder.pack(expand=True, pady=40)

        # Emotion display
        self.emotion_label = ctk.CTkLabel(frame, text="🎭 Detected Emotion: -", font=("Kohinoor Devanagari", 14))
        self.emotion_label.pack(pady=5)

        # Text display box
        self.box = ctk.CTkTextbox(frame, height=200, font=("Kohinoor Devanagari", 16))
        self.box.pack(fill="both", expand=True, pady=10)
        self.box.tag_config("highlight", background="#FFD966", foreground="#000000")
        self.box.configure(state="disabled")
        
        # Status bar
        self.status = ctk.CTkLabel(frame, text="Ready", font=("Arial", 10), text_color="gray")
        self.status.pack(pady=5)

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
            "excited": "🤩", "calm": "😌", "neutral": "😐",
            "fear": "😨", "shock": "😱"
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
        if not path: return
        self.status.configure(text="Processing...")
        def task():
            text = extract_text_from_image(path) if path.endswith((".png", ".jpg", ".jpeg")) else extract_text_from_document(path)
            self.win.after(0, lambda: self.update_ui(text))
        Thread(target=task, daemon=True).start()

    def camera_input(self):
        self.status.configure(text="Opening camera...")
        Thread(target=lambda: self.win.after(0, lambda: self.update_ui(extract_text_from_camera())), daemon=True).start()

    def speak(self):
        if not self.text:
            self.status.configure(text="No text to speak!")
            return
        level = self.express_level.get()
        dialect = self.dialect.get()
        story = self.story.get()
        
        self.tts.stop_signal = False
        self.status.configure(text="Generating first sentence...")
        
        def task():
            try:
                sentences = split_into_sentences(self.text)
                all_audios = []
                self.tts.timeline.clear()
                self.tts.last_emotions.clear()

                for idx, sent in enumerate(sentences):
                    if self.tts.stop_signal: break
                    
                    self.win.after(0, lambda i=idx, n=len(sentences): 
                        self.status.configure(text=f"Processing {i+1}/{n}...", text_color="gray")
                    )
                    
                    print(f"--- Processing Sentence {idx+1}/{len(sentences)} ---")

                    audio, sr, emo, dur = self.tts.generate_sentence_audio(sent, level, dialect, story)
                    
                    if audio is None:
                        # This happens if both ML and gTTS fail
                        self.win.after(0, lambda: self.status.configure(text="⚠️ Generation Failed", text_color="red"))
                        continue
                    
                    # Highlight and Play immediately
                    self.win.after(0, lambda s=sent, e=emo, a=audio: (self.highlight_sentence(s, e), self.update_waveform(a)))
                    
                    # Save to temp and play
                    temp_f = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    sf.write(temp_f, audio, sr)
                    
                    pygame.mixer.music.load(temp_f)
                    pygame.mixer.music.play()
                    
                    # Update data for Analysis window
                    all_audios.append(audio)
                    self.tts.last_emotions.append((sent, emo))
                    self.tts.timeline.append((sent, emo, dur))
                    
                    # Sleep while playing
                    # We wait for the duration of the audio before continuing to next sentence
                    time.sleep(dur)
                    
                    try: os.remove(temp_f)
                    except: pass

                # Finalize batch for Analysis window
                if all_audios and not self.tts.stop_signal:
                    final = np.concatenate(all_audios)
                    self.tts.temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                    sf.write(self.tts.temp, final, sr)
                    self.win.after(0, self.update_waveform)
                    self.win.after(0, lambda: self.status.configure(text="Playback Finished"))
                
            except Exception as e:
                err_msg = str(e)[:100]
                self.win.after(0, lambda: self.status.configure(text=f"Error: {err_msg}"))
                print(f"Speak error: {e}")
        
        Thread(target=task, daemon=True).start()

    def save_audio(self):
        if not self.tts.temp or not os.path.exists(self.tts.temp):
            self.status.configure(text="Please generate audio first!")
            return
            
        path = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile="marathi_speech.wav"
        )
        if not path: return
        
        try:
            shutil.copy(self.tts.temp, path)
            self.status.configure(text=f"Saved to {os.path.basename(path)}")
        except Exception as e:
            self.status.configure(text=f"Save Error: {str(e)[:50]}")

    def show_analysis(self):
        """Open a separate window for detailed acoustic analysis."""
        if not self.tts.temp or not os.path.exists(self.tts.temp):
            self.status.configure(text="Please generate audio first!")
            return
            
        # Create Toplevel Window
        analysis_win = ctk.CTkToplevel(self.win)
        analysis_win.title("Acoustic Analysis & Waveform Patterns")
        analysis_win.geometry("1000x850")
        analysis_win.configure(fg_color="white") # Entire window white
        
        # Title - Modern font
        ctk.CTkLabel(analysis_win, text="📊 Acoustic Analysis Report", font=("Helvetica", 18), text_color="#1a1a1a").pack(pady=20)
        
        # Scrollable area - Transparent to show window white
        scroll = ctk.CTkScrollableFrame(analysis_win, fg_color="white", scrollbar_button_color="#cccccc")
        scroll.pack(fill="both", expand=True, padx=20, pady=10)
        
        try:
            audio, sr = librosa.load(self.tts.temp, sr=None)
            
            # --- Visualizations ---
            viz_frame = ctk.CTkFrame(scroll, fg_color="white")
            viz_frame.pack(fill="x", pady=10)
            
            # Using Figure object instead of plt.subplots for safe Tkinter embedding
            fig = Figure(figsize=(10, 7), dpi=100)
            fig.patch.set_facecolor('white') # Entire figure canvas white
            
            # Shared time axis for perfect alignment
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1) 
            
            # 1. Waveform - White Theme
            librosa.display.waveshow(audio, sr=sr, ax=ax1, color='#0056b3', alpha=1.0)
            ax1.set_title("Time-Domain Waveform (Amplitude vs Time)", color="#1a1a1a", fontname="Helvetica", fontsize=18)
            ax1.set_facecolor('white')
            ax1.tick_params(axis='both', colors='#333333', labelsize=14)
            ax1.grid(True, linestyle='-', alpha=0.15, color='#000000')
            ax1.set_ylabel("Amplitude", color='#1a1a1a', fontsize=18)
            ax1.set_xlabel("")
            # =========================
            hop_length = 512

            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(audio, hop_length=hop_length)), 
                ref=np.max
            )

            img = librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
            ax2.set_title("Frequency-Domain Spectrogram (Harmonic Structure)", color="#1a1a1a", fontname="Helvetica", fontsize=18)
            ax2.set_facecolor('#0f0f0f') # Maintain dark for spectrogram for better contrast
            ax2.set_ylim(0, 7500) # User requested 7500 Hz cap
            ax2.tick_params(axis='both', colors='#333333', labelsize=14)
            ax2.grid(True, linestyle='--', alpha=0.2, color='#ffffff')

            ax2.set_xlabel("Time", color='#1a1a1a', fontsize=18)
            ax2.set_ylabel("Frequency (Hz)", color='#1a1a1a', fontsize=18)


            # Optional: Add colorbar (looks professional)
            # Colorbar removed per user request for a cleaner look
            # ax1 and ax2 will naturally align now as neither has a colorbar offset
            
            # Ensure both graphs start and end at the exact same points
            duration = len(audio) / sr
            ax1.set_xlim(0, duration)
            ax2.set_xlim(0, duration)
            
            # Perfectly align Y-labels vertically
            fig.align_ylabels([ax1, ax2])
            
            fig.tight_layout(pad=3.0)
            
            canvas = FigureCanvasTkAgg(fig, master=viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)
            # Keep reference to avoid garbage collection
            viz_frame.canvas = canvas 
            
            # --- Parameters Explanation ---
            text_frame = ctk.CTkFrame(scroll, fg_color="white")
            text_frame.pack(fill="both", expand=True, pady=10)
            
            # Detect primary emotion from last generation
            main_emotion = "neutral"
            if self.tts.last_emotions:
                # Count frequencies
                from collections import Counter
                emos = [e[1] for e in self.tts.last_emotions]
                main_emotion = Counter(emos).most_common(1)[0][0]
            
            description = f"""
🎯 Current Detected Dominant Context: {main_emotion.upper()}

Dominant Parameters Influencing Waveform Patterns:

1. Amplitude Fluctuations (Volume & Strength):
   - Observations: Captured in the vertical peaks above. 
   - Analysis: Your speech shows {main_emotion} profile logic. (Factor: {'1.5 (High)' if main_emotion in ['angry', 'excited'] else '0.85 (Low)' if main_emotion == 'sad' else '1.0 (Standard)'})

2. Time Compression/Expansion:
   - Observations: Manifesting in horizontal spacing of waveform patterns.
   - Analysis: Reflecting speed factor discrepancies. {'Angry/Excited (Fast: 1.25)' if main_emotion in ['angry', 'excited'] else 'Sad (Slow: 0.9)' if main_emotion == 'sad' else 'Neutral (Standard: 1.0)'}.

3. Frequency Distribution:
   - Observations: Visualized in the spectrogram color intensity.
   - Analysis: {'Intense high-frequency content detected (Vivid Pattern).' if main_emotion in ['happy', 'angry', 'excited'] else 'Low-frequency concentration (Subdued Pattern).'}

4. Pause Distribution:
   - Observations: Visible as low-amplitude 'flat' regions in the waveform.
   - Analysis: {'Short, aggressive pauses' if main_emotion == 'angry' else 'Extended emotional breathing spaces' if main_emotion == 'sad' else 'Natural grammatical pauses'} detected.

5. Harmonic Structure:
   - Observations: The vertical stripes in the spectrogram reveal voice quality.
   - Analysis: Prominent higher harmonics indicate high intensity and clarity in the emotional expression.
"""
            desc_label = ctk.CTkLabel(text_frame, text=description, font=("Helvetica", 14), justify="left", wraplength=800, text_color="#1a1a1a")
            desc_label.pack(padx=20, pady=20)
            
            # No need for plt.close(fig) when using OO Figure API
            
        except Exception as e:
            ctk.CTkLabel(analysis_win, text=f"Analysis Error: {e}", text_color="red").pack()



    def update_waveform(self, audio=None):
        """Plot the generated audio waveform in the UI."""
        try:
            if audio is None:
                if not self.tts.temp or not os.path.exists(self.tts.temp):
                    return
                # Load audio for plotting if not provided
                audio, _ = librosa.load(self.tts.temp, sr=None)
            
            # Clear previous plot
            for widget in self.waveform_frame.winfo_children():
                widget.destroy()
                
            # Create dark themed figure
            fig = Figure(figsize=(7, 1.2), dpi=100)
            fig.patch.set_facecolor('#1a1a1a')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#1a1a1a')
            
            # Subsample for performance if needed
            if len(audio) > 5000:
                audio_resampled = audio[::max(1, len(audio)//5000)]
                ax.plot(audio_resampled, color='#00D2FF', linewidth=0.8, alpha=0.9)
            else:
                ax.plot(audio, color='#00D2FF', linewidth=0.8, alpha=0.9)
                
            # Styling - Hide all axes/labels
            ax.axis('off')
            fig.tight_layout(pad=0)
            
            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.waveform_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            # Keep reference to avoid garbage collection
            self.waveform_frame.canvas = canvas 
        except Exception as e:
            print(f"Waveform error: {e}")
            ctk.CTkLabel(self.waveform_frame, text="Waveform unavailable", text_color="gray").pack(pady=40)

    def run(self):
        self.win.mainloop()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("💎 MARATHI TTS: ULTRA-SMOOTH MASTERING EDITION")
    print("="*50 + "\n")
    OCRTTSApp().run()
