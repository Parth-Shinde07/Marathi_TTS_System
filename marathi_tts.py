# OCR → Marathi Text → Speech (USER-FRIENDLY + PROSODY)

# pip install gtts pygame customtkinter pillow pytesseract opencv-python numpy scipy soundfile

import customtkinter as ctk
from gtts import gTTS
import pygame
import os
import soundfile as sf
from tkinter import filedialog
from threading import Thread
import numpy as np
from scipy import signal
import tempfile
import pytesseract
from PIL import Image

pygame.mixer.init()

class ProsodyModifier:
    def apply(self, audio, level, emotion):
        # Base from slider
        speed = 0.9 + (level / 100) * 0.3
        intensity = 0.9 + (level / 100) * 0.8
        volume = 0.85 + (level / 100) * 0.5

        # Emotion-specific adjustment
        if emotion == "happy":
            speed *= 1.1
            intensity *= 1.2
        elif emotion == "angry":
            speed *= 1.15
            intensity *= 1.4
            volume *= 1.2
        elif emotion == "sad":
            speed *= 0.85
            intensity *= 0.7
            volume *= 0.8

        # Apply rhythm
        audio = signal.resample(audio, int(len(audio) / speed))

        # Apply intensity
        mean = np.mean(audio)
        audio = mean + (audio - mean) * intensity

        return audio * volume


# -------------------- Marathi Dialect --------------------
class MarathiDialect:
    def __init__(self, rules):
        self.rules = rules

    def apply(self, text):
        for k, v in self.rules.items():
            text = text.replace(k, v)
        return text

def detect_emotion_from_text(text):
    text = text.strip()

    if "!" in text or any(w in text for w in ["आनंद", "खूप छान", "यश"]):
        return "happy"

    if any(w in text for w in ["दुःख", "वाईट", "कष्ट", "त्रास"]):
        return "sad"

    if any(w in text for w in ["राग", "नको", "चिड", "बस"]):
        return "angry"

    return "neutral"

def add_emotion_pauses(text, emotion):
    if emotion == "sad":
        text = text.replace("।", "… ")
    elif emotion == "angry":
        text = text.replace("।", "! ")
    return text


# -------------------- Marathi TTS --------------------
class MarathiTTS:
    def __init__(self):
        self.prosody = ProsodyModifier()
        self.temp_file = None
        self.is_playing = False

        # Marathi regional dialect rules
        self.dialects = {
            "Standard": MarathiDialect({}),
            "Varhadi": MarathiDialect({
                "आहे": "आय", "नाही": "नाय", "मला": "म्हाला"
            }),
            "Ahirani": MarathiDialect({
                "आहे": "हाय", "नाही": "नाय", "मला": "म्हाला"
            }),
            "Malwani": MarathiDialect({
                "आहे": "आस", "नाही": "नाय", "मला": "मका"
            })
        }

    def generate(self, text, prosody_level, dialect):
        # detect emotion
        emotion = detect_emotion_from_text(text)

        # apply dialect
        text = self.dialects[dialect].apply(text)

        # apply emotion pauses
        text = add_emotion_pauses(text, emotion)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav_path = tmp.name
        mp3_path = wav_path.replace(".wav", ".mp3")
        tmp.close()

        gTTS(text=text, lang="mr").save(mp3_path)
        audio, sr = sf.read(mp3_path)

        # apply emotion-aware prosody
        audio = self.prosody.apply(audio, prosody_level, emotion)

        sf.write(wav_path, audio, sr)
        os.remove(mp3_path)

        self.temp_file = wav_path


    def play(self):
        if self.temp_file:
            pygame.mixer.music.load(self.temp_file)
            pygame.mixer.music.play()
            self.is_playing = True

    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False



# -------------------- OCR --------------------
def extract_text_from_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang="mar").strip()


# -------------------- GUI --------------------
class OCRTTSApp:
    def __init__(self):
        self.tts = MarathiTTS()
        self.text = ""

        ctk.set_appearance_mode("light")
        self.win = ctk.CTk()
        self.win.title("OCR आधारित मराठी TTS")
        self.win.geometry("700x550")

        self.frame = ctk.CTkFrame(self.win)
        self.frame.pack(expand=True, fill="both", padx=20, pady=20)

        ctk.CTkLabel(
            self.frame,
            text="OCR → मराठी टेक्स्ट-टू-स्पीच",
            font=("Arial", 22, "bold")
        ).pack(pady=10)

        ctk.CTkButton(
            self.frame,
            text="📷 Marathi Image निवडा",
            font=("Arial", 15),
            height=45,
            command=self.select_image
        ).pack(pady=10)

        # OCR text display
        ctk.CTkLabel(self.frame, text="OCR Extracted Text").pack(pady=(10, 5))
        self.ocr_textbox = ctk.CTkTextbox(self.frame, height=300)
        self.ocr_textbox.pack(fill="x", padx=20)
        self.ocr_textbox.configure(state="disabled")

        # Emotion display
        self.emotion_label = ctk.CTkLabel(
            self.frame, text="Detected Emotion: -", font=("Arial", 14, "bold")
        )
        self.emotion_label.pack(pady=8)

        # Prosody slider
        ctk.CTkLabel(self.frame, text="Prosody Control").pack(pady=5)
        self.prosody_slider = ctk.CTkSlider(
            self.frame, from_=0, to=100, number_of_steps=100
        )
        self.prosody_slider.set(50)
        self.prosody_slider.pack(fill="x", padx=30)

        # Dialect selection
        ctk.CTkLabel(self.frame, text="Marathi Regional Language").pack(pady=5)
        self.dialect_var = ctk.StringVar(value="Standard")
        ctk.CTkOptionMenu(
            self.frame,
            values=["Standard", "Varhadi", "Ahirani", "Malwani"],
            variable=self.dialect_var
        ).pack(pady=5)

        # Buttons
        btn_frame = ctk.CTkFrame(self.frame)
        btn_frame.pack(pady=20, fill="x")

        ctk.CTkButton(
            btn_frame, text="▶ Run / Speak", command=self.run_tts
        ).pack(side="left", expand=True, padx=10)

        ctk.CTkButton(
            btn_frame, text="⏹ Stop", command=self.stop_tts
        ).pack(side="right", expand=True, padx=10)

        self.status = ctk.CTkLabel(self.frame, text="")
        self.status.pack(pady=10)

    def safe_ui_update(self, func):
        self.win.after(0, func)

    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if not path:
            return

        self.status.configure(text="OCR चालू आहे...")

        def task():
            text = extract_text_from_image(path)
            emotion = detect_emotion_from_text(text) if text else "Neutral"

            def update_ui():
                if text:
                    self.text = text
                    self.ocr_textbox.configure(state="normal")
                    self.ocr_textbox.delete("1.0", "end")
                    self.ocr_textbox.insert("end", text)
                    self.ocr_textbox.configure(state="disabled")
                    self.emotion_label.configure(
                        text=f"Detected Emotion: {emotion}"
                    )
                    self.status.configure(text="OCR पूर्ण – Run दाबा")
                else:
                    self.status.configure(text="मजकूर सापडला नाही")

            self.safe_ui_update(update_ui)

        Thread(target=task, daemon=True).start()

    def run_tts(self):
        if not self.text:
            self.status.configure(text="आधी Image निवडा")
            return

        self.status.configure(text="बोलणे सुरू आहे...")
        level = self.prosody_slider.get()
        dialect = self.dialect_var.get()

        def task():
            self.tts.generate(self.text, level, dialect)
            self.tts.play()

        Thread(target=task, daemon=True).start()

    def stop_tts(self):
        self.tts.stop()
        self.status.configure(text="बोलणे थांबवले")

    def run(self):
        self.win.mainloop()



# -------------------- Main --------------------
if __name__ == "__main__":
    OCRTTSApp().run()
