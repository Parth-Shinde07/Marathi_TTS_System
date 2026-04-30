import os
import librosa
import soundfile as sf
import pyloudnorm as pyln
import numpy as np
import noisereduce as nr

INPUT_DIR = "raw_wavs"
OUTPUT_DIR = "wavs"
TARGET_SR = 22050
TARGET_LUFS = -23.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
meter = pyln.Meter(TARGET_SR)

def process_audio(file_path, output_path):
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=None)
    
    # 2. Resample if necessary
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    
    # 3. Noise Reduction (using first 0.1s as noise profile)
    y_clean = nr.reduce_noise(y=y, sr=TARGET_SR, y_noise=y[:int(TARGET_SR*0.1)], prop_decrease=0.8)
    
    # 4. Strip silence at beginning and end
    y_trimmed, _ = librosa.effects.trim(y_clean, top_db=30)
    
    # 5. Volume Normalization (EBU R128)
    loudness = meter.integrated_loudness(y_trimmed)
    y_norm = pyln.normalize.loudness(y_trimmed, loudness, TARGET_LUFS)
    
    # 6. Save as 16-bit PCM WAV
    sf.write(output_path, y_norm, TARGET_SR, subtype='PCM_16')

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"Please create {INPUT_DIR} and place your audio files there.")
    else:
        for f in os.listdir(INPUT_DIR):
            if f.endswith(".wav"):
                process_audio(os.path.join(INPUT_DIR, f), os.path.join(OUTPUT_DIR, f))
        print("Audio processing complete.")
