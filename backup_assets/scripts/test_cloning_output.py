import torch
import os
import soundfile as sf
import numpy as np

# Apply safety fixes
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from coqpit import Coqpit
try:
    from TTS.tts.layers.xtts.gpt import GPTArgs
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, Coqpit, XttsArgs, GPTArgs])
except ImportError:
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, Coqpit, XttsArgs])

from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"
device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

speaker_wav = "/Users/parth/Documents/Marathi speech Database Sarang Joshi/prosodyrich_137.wav"
text = "नमस्कार, आपण कसे आहात?"
output_file = "cloning_test.wav"

print(f"Loading XTTS v2 on {device}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print(f"Generating with speaker_wav: {speaker_wav}")
try:
    waveform = tts.tts(
        text=text,
        language="hi",
        speaker_wav=speaker_wav,
        speed=1.0
    )
    sf.write(output_file, waveform, 24000)
    print(f"Test complete. Output saved to {output_file}")
except Exception as e:
    print(f"Generation Error: {e}")
