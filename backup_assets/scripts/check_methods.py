import torch
import os
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

print(f"Loading XTTS v2 on {device}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print("Checking methods...")
model = tts.synthesizer.tts_model
print("Has get_conditioning_latents:", hasattr(model, "get_conditioning_latents"))
