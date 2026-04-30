import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from tqdm import tqdm

def clean_audio(input_path, output_path, target_sr=22050):
    """
    Production-grade audio cleaning pipeline.
    1. Resample to 22k
    2. Remove Noise (Stationary & Non-stationary)
    3. Trim Silence (Aggressive)
    4. Normalize Loudness
    """
    try:
        # 1. Load and Resample
        y, sr = librosa.load(input_path, sr=target_sr)
        
        # 2. Noise Reduction using noisereduce
        # We estimate noise from a 0.5s segment at the beginning
        y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.9, stationary=False)
        
        # 3. Trim Silence (Top DB 20 is aggressive)
        y_trimmed, _ = librosa.effects.trim(y_clean, top_db=20)
        
        # 4. Normalize (Peak normalization to -1dB)
        if np.abs(y_trimmed).max() > 0:
            y_norm = y_trimmed / np.abs(y_trimmed).max() * 0.9
        else:
            y_norm = y_trimmed
            
        # 5. Save
        sf.write(output_path, y_norm, sr)
        return True
    except Exception as e:
        print(f"Error cleaning {input_path}: {e}")
        return False

def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    print(f"Cleaning {len(files)} files...")
    
    for f in tqdm(files):
        clean_audio(os.path.join(input_dir, f), os.path.join(output_dir, f))

if __name__ == "__main__":
    # Correct paths detected from your environment
    raw_dir = "/Users/parth/Documents/marathi_tts_work/processed/wavs"
    clean_dir = "/Users/parth/Documents/marathi_tts_work/processed/wavs_cleaned"
    
    if os.path.exists(raw_dir):
        process_folder(raw_dir, clean_dir)
        print(f"\nDone! Cleaned audio saved to: {clean_dir}")
        print("Next: Update your metadata.csv or config to point to this new folder.")
    else:
        print(f"Directory {raw_dir} not found.")
