import sys
import os
import torch
import torchaudio

# Add local TTS clone to path
sys.path.append(os.path.join(os.getcwd(), "TTS"))

# Force torchaudio to NOT use torchcodec
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

# Monkeypatch torchaudio.load
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

# Monkeypatch Coqui Trainer for MPS support
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

trainer.generic_utils.to_cuda = mps_to_cuda
trainer.generic_utils.get_cuda = mps_get_cuda

if not hasattr(trainer.trainer_utils, "setup_torch_training_env_orig"):
    trainer.trainer_utils.setup_torch_training_env_orig = trainer.trainer_utils.setup_torch_training_env
trainer.trainer_utils.setup_torch_training_env = mps_setup_torch_training_env

_orig_cuda = torch.nn.Module.cuda
def mps_cuda_patch(self, device=None):
    if torch.backends.mps.is_available():
        return self.to("mps")
    return _orig_cuda(self, device)
torch.nn.Module.cuda = mps_cuda_patch

print(" [!] MPS Monkeypatch applied successfully.")

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits

def train():
    # Paths
    config_path = "/Users/parth/Documents/Marathi_TTS_System/vits_marathi_config_30k.json"
    output_path = "/Users/parth/Documents/Marathi_TTS_System/tts_30k_output"
    
    # Load config
    config = VitsConfig()
    config.load_json(config_path)
    
    # Update output path just in case
    config.output_path = output_path

    # Override to enforce 30,000-step training at LR = 5e-5
    config.lr = 5e-5
    config.lr_gen = 5e-5
    config.lr_disc = 5e-5
    print(f"[INFO] Training for 30,000 steps at LR={config.lr}")
    
    # Load samples
    # The ljspeech formatter in Coqui TTS handles the dataset path from config
    train_samples, eval_samples = load_tts_samples(
        config.datasets[0],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    # Initialize model
    model = Vits.init_from_config(config)
    
    # Initialize trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Start training
    trainer.fit()

if __name__ == "__main__":
    train()
