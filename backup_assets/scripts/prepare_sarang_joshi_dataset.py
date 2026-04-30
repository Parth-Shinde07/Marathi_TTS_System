import os
import torch
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import shutil

# Configuration
SOURCE_DIR = "/Users/parth/Documents/Marathi speech Database Sarang Joshi"
TARGET_DIR = "/Users/parth/Documents/sarang_joshi_processed"
WAVS_SUBDIR = os.path.join(TARGET_DIR, "wavs")
METADATA_FILE = os.path.join(TARGET_DIR, "metadata.csv")

# Create directories
os.makedirs(WAVS_SUBDIR, exist_ok=True)

# Device selection
device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize ASR pipeline
print("Initializing ASR pipeline...")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device
)

# Find all wav files
print(f"Searching for wav files in {SOURCE_DIR}...")
wav_files = []
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith(".wav"):
            wav_files.append(os.path.join(root, file))

print(f"Found {len(wav_files)} files.")

# Transcription and processing
metadata = []
print("Transcribing files (this may take a few minutes)...")

# We'll limit to first 100 files for now to show progress and avoid taking too long in one go
# Or should we do all? 835 isn't that many. Let's do all with a nice progress bar.
for wav_path in tqdm(wav_files):
    file_name = os.path.basename(wav_path)
    target_wav_path = os.path.join(WAVS_SUBDIR, file_name)
    
    # Symlink the wav file to our processed directory
    if not os.path.exists(target_wav_path):
        os.symlink(wav_path, target_wav_path)
    
    try:
        # Transcribe
        result = asr(wav_path, generate_kwargs={"language": "marathi"})
        transcription = result["text"].strip()
        
        # Append to metadata
        metadata.append({
            "file_name": f"wavs/{file_name}",
            "transcription": transcription,
            "label": "neutral"
        })
    except Exception as e:
        print(f"Error transcribing {file_name}: {e}")

# Save metadata to CSV
df = pd.DataFrame(metadata)
df.to_csv(METADATA_FILE, index=False)

print(f"\n[DONE] Dataset prepared at {TARGET_DIR}")
print(f"Metadata saved to {METADATA_FILE}")
print(f"Total files transcribed: {len(metadata)}")
