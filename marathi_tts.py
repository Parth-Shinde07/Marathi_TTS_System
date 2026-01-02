# OCR → Marathi Text → Speech (USER-FRIENDLY + PROSODY)

import customtkinter as ctk
from gtts import gTTS
import pygame, os, re, tempfile
import soundfile as sf
import numpy as np
from scipy import signal
from threading import Thread
from tkinter import filedialog
from PIL import Image
import pytesseract
import pdfplumber
import cv2

pygame.mixer.init()

# -------------------- Prosody --------------------
class ProsodyModifier:
    def apply(self, audio, level, emotion, voice=None):
        speed = 0.9 + (level / 100) * 0.3
        intensity = 0.9 + (level / 100) * 0.8
        volume = 0.85 + (level / 100) * 0.5

        if emotion == "neutral":
            speed *= 0.95
            intensity *= 0.9
            volume *= 0.9
        elif emotion == "happy":
            speed *= 1.2
        elif emotion == "sad":
            speed *= 0.85
            intensity *= 0.7
        elif emotion == "angry":
            speed *= 1.15
            intensity *= 1.4
            volume *= 1.4

        if voice:
            speed *= voice["speed"]
            intensity *= voice["intensity"]
            volume *= voice["volume"]

        audio = signal.resample(audio, int(len(audio) / speed))
        mean = np.mean(audio)
        audio = mean + (audio - mean) * intensity
        return audio * volume


# -------------------- Emotion --------------------
EMOTION_KEYWORDS = {
    "happy": ["आनंद", "छान", "मस्त", "यश", "खुश"],
    "sad": ["दुःख", "वाईट", "कष्ट", "त्रास", "ताण"],
    "angry": ["राग", "चिड", "नको"]
}

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def detect_emotion(text, story=False):
    if story:
        return "neutral"
    words = tokenize(text)
    scores = {k: 0 for k in EMOTION_KEYWORDS}
    for e, keys in EMOTION_KEYWORDS.items():
        for k in keys:
            if k in words:
                scores[e] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] else "neutral"


# -------------------- Voices --------------------
VOICES = {
    "narration": {"speed": 0.95, "intensity": 0.9, "volume": 0.9},
    "dialogue": {"speed": 1.05, "intensity": 1.0, "volume": 1.05},
}


def split_story(text):
    lines = text.split("\n")
    segments = []
    for l in lines:
        l = l.strip()
        if not l:
            continue
        if l.startswith(("\"", "“")):
            segments.append((l.strip("“”\""), "dialogue"))
        else:
            segments.append((l, "narration"))
    return segments


# -------------------- OCR --------------------
def extract_text_from_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang="mar").strip()

def extract_text_from_document(path):
    text = ""
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif path.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""
    return text.strip()

def extract_text_from_camera():
    cap = cv2.VideoCapture(0)
    text = ""
    while True:
        ret, frame = cap.read()
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
    return text.strip()


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
    "Varhadi": MarathiDialect({"आहे": "आय", "नाही": "नाय", "मी": "म्ही",
                               'गा': 'मा', 'ळ': 'ल',
                               'काय': 'काय', 'तू': 'तु',
                               'आपण': 'आपुण', 'झाला': 'झाला',
                               'पाकहजे': 'पाकहजे', 'बोलतो': 'बोलतो'}),

    "Malvani": MarathiDialect({"आहे": "आसा", "नाही": "ना", "मला": "माका",
                               'व': 'व्ह', 'च': 'च', 'झ': 'झ', 'आहे': 'आस',
                               'नाही': 'नाय', 'काय': 'काय',
                               'कसं': 'कसं', 'तुला': 'तुज्जा',
                               'मला': 'मज्जा', 'पाकहजे': 'पायजे'}),

    "Ahirani": MarathiDialect({"आहे": "हाय", "नाही": "नाय",
                               'आहे': 'हाय', 'नाही': 'नाय', 'मला': 'म्हाला',
                               'तुला': 'तुला', 'झाला': 'झालं',
                               'काय': 'काय', 'कसं': 'कसं',
                               'पाकहजे': 'पायजे', 'जातो': 'जातो'}),

    "Kokani": MarathiDialect({'आहे': 'आसा', 'नाही': 'ना', 'काय': 'ककतं',
                              'कसं': 'कसं', 'तुला': 'तुका',
                              'मला': 'माका', 'पाकहजे': 'जाय'})
}


# -------------------- TTS --------------------
class MarathiTTS:
    def __init__(self):
        self.prosody = ProsodyModifier()
        self.temp = None

    def generate(self, text, level, dialect, story):
        text = DIALECTS[dialect].apply(text)
        segments = split_story(text)
        audios = []

        for seg, seg_type in segments:
            emotion = detect_emotion(seg, story)
            voice = VOICES["dialogue"] if seg_type == "dialogue" else VOICES["narration"]

            mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            wav = mp3.replace(".mp3", ".wav")

            gTTS(seg, lang="mr").save(mp3)
            audio, sr = sf.read(mp3)
            audio = self.prosody.apply(audio, level, emotion, voice)

            audios.append(audio)
            os.remove(mp3)

        final = np.concatenate(audios)
        self.temp = wav
        sf.write(wav, final, sr)

    def play(self):
        pygame.mixer.music.load(self.temp)
        pygame.mixer.music.play()

    def stop(self):
        pygame.mixer.music.stop()


# -------------------- GUI --------------------
class OCRTTSApp:
    def __init__(self):
        self.tts = MarathiTTS()
        self.text = ""

        self.win = ctk.CTk()
        self.win.geometry("700x550")
        self.win.title("Marathi OCR TTS")

        frame = ctk.CTkFrame(self.win)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkButton(frame, text="📁 Image / Document", command=self.select_input).pack(pady=5)
        ctk.CTkButton(frame, text="📷 Camera OCR", command=self.camera_input).pack(pady=5)

        self.box = ctk.CTkTextbox(frame, height=250)
        self.box.pack(fill="x", pady=10)
        self.box.configure(state="disabled")

        self.story = ctk.BooleanVar()
        ctk.CTkCheckBox(frame, text="Story Mode", variable=self.story).pack()

        self.slider = ctk.CTkSlider(frame, from_=0, to=100)
        self.slider.set(50)
        self.slider.pack(fill="x", pady=10)

        self.dialect = ctk.StringVar(value="Standard")
        ctk.CTkOptionMenu(frame, values=list(DIALECTS.keys()), variable=self.dialect).pack()

        ctk.CTkButton(frame, text="▶ Speak", command=self.speak).pack(pady=5)
        ctk.CTkButton(frame, text="⏹ Stop", command=self.tts.stop).pack()

    def update_ui(self, text):
        self.text = text
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        self.box.insert("end", text)
        self.box.configure(state="disabled")

    def select_input(self):
        path = filedialog.askopenfilename(
            parent=self.win,
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("Docs", "*.pdf *.txt")]
        )
        if not path:
            return

        def task():
            text = extract_text_from_image(path) if path.endswith((".png", ".jpg", ".jpeg")) else extract_text_from_document(path)
            self.win.after(0, lambda: self.update_ui(text))

        Thread(target=task, daemon=True).start()

    def camera_input(self):
        Thread(
            target=lambda: self.win.after(0, lambda: self.update_ui(extract_text_from_camera())),
            daemon=True
        ).start()

    def speak(self):
        if not self.text:
            return

        level = self.slider.get()
        dialect = self.dialect.get()
        story = self.story.get()

        Thread(
            target=lambda: (
                self.tts.generate(self.text, level, dialect, story),
                self.tts.play()
            ),
            daemon=True
        ).start()

    def run(self):
        self.win.mainloop()


if __name__ == "__main__":
    OCRTTSApp().run()
