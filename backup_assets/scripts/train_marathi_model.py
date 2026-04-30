import os
import torch
from transformers import VitsModel, VitsTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Audio

import argparse

# Configuration for IIT Bombay dataset training
MODEL_ID = "facebook/mms-tts-mar"
DATASET_PATH = "/Users/parth/Documents/marathi_tts_work/processed" 
OUTPUT_DIR = "./marathi_human_model"

def train(max_steps=None):
    print("--- Marathi ML Model Training Pipeline ---")
    
    # 1. Load Pre-trained Marathi Backbone
    tokenizer = VitsTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID)

    # 1.1 Update Model Configuration with HiFi-GAN specifications
    # Applied lambdas (λ_mel = 45.0, λ_fm = 1.0)
    model.config.mel_loss_alpha = 45.0
    model.config.feat_loss_alpha = 1.0
    model.config.kl_loss_alpha = 1.0
    model.config.dur_loss_alpha = 1.0
    print(f"[INFO] Applied HiFi-GAN loss weights: λ_mel={model.config.mel_loss_alpha}, λ_fm={model.config.feat_loss_alpha}")

    # 2. Load Dataset manually to avoid torchcodec/ffmpeg issues with audiofolder
    import pandas as pd
    from datasets import Dataset, Features, Value, Audio as AudioFeature
    
    print(f"[INFO] Loading metadata from {DATASET_PATH}/metadata.csv...")
    df = pd.read_csv(os.path.join(DATASET_PATH, "metadata.csv"))
    
    # Prepend the directory path to file_name if not already correct
    # (The metadata already has 'wavs/' prepended from my previous fix)
    df["audio"] = df["file_name"].apply(lambda x: os.path.join(DATASET_PATH, x))
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", AudioFeature(sampling_rate=16000))
    dataset = dataset.train_test_split(test_size=0.1)
    
    # 3. Preprocessing
    def preprocess_function(batch):
        audio = batch["audio"]
        
        # Process text
        inputs = tokenizer(text=batch["transcription"], return_tensors="pt")
        batch["input_ids"] = inputs["input_ids"][0]
        
        # Process audio to 16kHz (standard for MMS)
        batch["labels"] = audio["array"]
        return batch

    print("[INFO] Preprocessing dataset...")
    # cast audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    processed_dataset = dataset.map(preprocess_function, remove_columns=dataset["train"].column_names)

    # 4. Data Collator
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorForVits:
        tokenizer: VitsTokenizer

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            # Extract inputs and labels
            input_ids = [feature["input_ids"] for feature in features]
            labels = [feature["labels"] for feature in features]

            # Pad text inputs
            batch = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
            )

            # Pad audio labels
            # Calculate max length in batch
            max_label_len = max(len(l) for l in labels)
            
            # Create tensor with zero padding
            padded_labels = torch.zeros((len(labels), max_label_len), dtype=torch.float32)
            for i, label in enumerate(labels):
                padded_labels[i, :len(label)] = torch.tensor(label, dtype=torch.float32)
            
            batch["labels"] = padded_labels
            return batch

    # 5. Training Arguments
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    
    print(f"[INFO] Using device: {device}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, # Increased for MPS efficiency
        gradient_accumulation_steps=8,  # Adjusted to keep effective batch size at ~32
        learning_rate=5e-5,            # Slow, stable LR for 30k-step training run
        num_train_epochs=50,           # Increased from 10 to 50
        warmup_steps=500,              # Added warmup for stability
        save_steps=1000,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=3,            # Keep only last 3 checkpoints
        load_best_model_at_end=True,   # Load best model at end
        metric_for_best_model="loss",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        use_mps_device=torch.backends.mps.is_available(),
        max_steps=30000,               # Fixed target: 30,000 training steps
    )

    # 6. Initialize Trainer
    data_collator = DataCollatorForVits(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n[READY] Starting training. This may take several hours...")
    trainer.train()
    
    # Save the final best model explicitly
    print(f"[INFO] Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    train(max_steps=args.max_steps)
