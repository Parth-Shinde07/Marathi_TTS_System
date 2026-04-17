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

print(f"Loading XTTS v2 on {device}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

print("Attempting to get conditioning latents manually...")
try:
    model = tts.synthesizer.tts_model
    # Most XTTS models expect audio_path as a list
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
    print("SUCCESS: Latents calculated.")
    
    print("Attempting generation with pre-calculated latents...")
    # Generate using the internal model directly to avoid the API wrapper's torchcodec check
    out = model.inference(
        text="नमस्कार, आपण कसे आहात?",
        language="hi",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7,
    )
    waveform = out["wav"].cpu().numpy()
    sf.write("bypass_test.wav", waveform, 24000)
    print("SUCCESS: Audio generated and saved to bypass_test.wav")

except Exception as e:
    print(f"Bypass Failed: {e}")
    import traceback
    traceback.print_exc()
