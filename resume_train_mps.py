import os
import sys
import torch
import torchaudio

# Ensure local TTS is in path
sys.path.append(os.path.join(os.getcwd(), "TTS"))

# 1. Force torchaudio to NOT use torchcodec
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

# 2. Monkeypatch torchaudio.load
_orig_load = torchaudio.load
def safe_load(filepath, **kwargs):
    try:
        return _orig_load(filepath, **kwargs)
    except Exception as e:
        if "TorchCodec" in str(e):
            import soundfile as sf
            data, sr = sf.read(filepath)
            return torch.from_numpy(data).float().unsqueeze(0), sr
        raise e
torchaudio.load = safe_load

# 3. Monkeypatch Coqui Trainer for MPS support
import trainer.generic_utils
import trainer.trainer_utils

def mps_to_cuda(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.contiguous()
        if torch.backends.mps.is_available():
            return x.to("mps")
        if torch.cuda.is_available():
            return x.cuda(non_blocking=True)
    return x

def mps_get_cuda():
    if torch.backends.mps.is_available():
        return True, torch.device("mps")
    return torch.cuda.is_available(), torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mps_setup_torch_training_env(*args, **kwargs):
    if torch.backends.mps.is_available():
        torch.manual_seed(kwargs.get("training_seed", 54321))
        return True, 1
    return trainer.trainer_utils.setup_torch_training_env_orig(*args, **kwargs)

# Apply monkeypatches
trainer.generic_utils.to_cuda = mps_to_cuda
trainer.generic_utils.get_cuda = mps_get_cuda

if not hasattr(trainer.trainer_utils, "setup_torch_training_env_orig"):
    trainer.trainer_utils.setup_torch_training_env_orig = trainer.trainer_utils.setup_torch_training_env
trainer.trainer_utils.setup_torch_training_env = mps_setup_torch_training_env

# 4. Monkeypatch nn.Module.cuda for MPS
_orig_cuda = torch.nn.Module.cuda
def mps_cuda_patch(self, device=None):
    if torch.backends.mps.is_available():
        return self.to("mps")
    return _orig_cuda(self, device)
torch.nn.Module.cuda = mps_cuda_patch

print(" [!] MPS Monkeypatch applied successfully for Resume.")

from TTS.bin.train_tts import main

if __name__ == "__main__":
    # Path to the latest output directory containing checkpoint_100.pth
    CONTINUE_PATH = "/Users/parth/Documents/Marathi_TTS_System/tts_30k_output/vits_marathi_30k-April-26-2026_11+57AM-c3ed134"
    
    sys.argv = [
        "train_tts", 
        "--continue_path", CONTINUE_PATH
    ]
    
    print(f" [!] Resuming training from: {CONTINUE_PATH}")
    main()
