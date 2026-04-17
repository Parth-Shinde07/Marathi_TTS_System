import torch
from transformers import pipeline
import os

# Device selection
device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Path to the file
file_path = "/Users/parth/Documents/Marathi speech Database Sarang Joshi/prosodyrich_01.wav"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

print(f"Transcribing {os.path.basename(file_path)}...")

# Initialize ASR pipeline
# Using whisper-small for a good balance of speed and precision in Marathi
try:
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device
    )

    # Transcribe
    result = asr(file_path, generate_kwargs={"language": "marathi"})
    print("\n--- Transcription ---")
    print(result["text"])
    print("----------------------")
except Exception as e:
    print(f"Error during ASR: {e}")
