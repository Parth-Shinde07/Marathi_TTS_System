import os
import sys

# Force torchaudio to NOT use torchcodec
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0" 

import torchaudio

# Monkeypatch torchaudio.load to avoid torchcodec
_orig_load = torchaudio.load
def safe_load(filepath, **kwargs):
    # Force 'ffmpeg' or 'soundfile' by passing backend if supported in this version
    # fallback to original but try to avoid the dispatcher logic if possible
    try:
        return _orig_load(filepath, **kwargs)
    except Exception as e:
        if "TorchCodec" in str(e):
            # Try to use a different backend directly
            # This is a bit hacky but might work if we can find another loader
            import soundfile as sf
            import torch
            data, sr = sf.read(filepath)
            return torch.from_numpy(data).float().unsqueeze(0), sr
        raise e

torchaudio.load = safe_load

# Ensure TTS is in path
from TTS.bin.train_tts import main

if __name__ == "__main__":
    # Remove wrapper name and set actual args
    sys.argv = ["train_tts", "--config_path", "vits_marathi_config_30k.json"]
    main()
