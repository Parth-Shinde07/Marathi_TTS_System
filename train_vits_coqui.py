import os
from trainer import Trainer, TrainingArgs
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
    model = Vits(config)
    
    # Initialize trainer
    trainer = Trainer(
        TrainingArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    train()
