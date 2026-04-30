import torch
import numpy as np
import os
import re
from transformers import VitsModel, VitsTokenizer
import librosa

class MMSVitsEngine:
    """
    Standard Marathi TTS using VITS (based on IIT Bombay/Meta findings).
    Highly reliable and natural for Marathi.

    TECHNICAL SPECIFICATION (HiFi-GAN Vocoder):
    - Generator (G): Creates audio from Mel-spectrogram input (**80 Mel filters**).
    - Discriminator (D): Multi-Period & Multi-Scale Discriminators for realism.
    
    LOSS FUNCTIONS:
    L_total = L_adv + λ_fm * L_fm + λ_mel * L_mel
    
    1. Adversarial Loss (L_adv):
       E[(D(real) - 1)^2] + E[D(G(spec))^2]
    
    2. Feature Matching Loss (L_fm):
       Σ ||D_layer(real) - D_layer(G(spec))|| 
       (Priority: Natural quality, Tone, Texture)
    
    3. Mel-Spectrogram Loss (L_mel):
       ||Mel(real) - Mel(G(spec))||
       (Priority: Accuracy, Pronunciation, Content)

    LAMBDA SETTINGS (Current):
    - λ_mel = 45.0 (High focus on pronunciation accuracy)
    - λ_fm  = 1.0  (Standard focus on natural textures)
    """
    def __init__(self, model_id="facebook/mms-tts-mar"):
        # Check if fine-tuned model exists, otherwise use base
        local_path = "./marathi_human_model"
        if os.path.exists(local_path) and any(os.listdir(local_path)):
            print(f"Loading Fine-tuned model from {local_path}...")
            model_id = local_path
            
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Loading VITS Model ({model_id}) on {self.device}...")
        try:
            self.tokenizer = VitsTokenizer.from_pretrained(model_id)
            self.model = VitsModel.from_pretrained(model_id).to(self.device)
            self.sampling_rate = self.model.config.sampling_rate
            print("VITS Model loaded successfully.")
        except Exception as e:
            print(f"Error loading VITS model: {e}")
            # Fallback to base model if local load fails
            if model_id == local_path:
                print("Local load failed, falling back to base facebook/mms-tts-mar")
                self.tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-mar")
                self.model = VitsModel.from_pretrained("facebook/mms-tts-mar").to(self.device)
                self.sampling_rate = self.model.config.sampling_rate

    def generate_speech(self, text, speed=1.0, **kwargs):
        """MMS/VITS generation with quality-focused parameters."""
        # Strip newlines and non-Devanagari characters to ensure the MMS tokenizer doesn't hallucinate
        text = text.replace("\n", " ")
        text = re.sub(r"[^\u0900-\u097F ।\.!?]", "", text)
        
        # Move model to CPU for inference to resolve the 'leaky_relu' MPS/Metal bug
        # VITS is highly optimized and runs in real-time even on a single CPU core.
        self.model.to("cpu")
        inputs = self.tokenizer(text=text, return_tensors="pt").to("cpu")
        
        # Set to absolute deterministic mode for the purest possible signal
        self.model.speaking_rate = speed
        self.model.noise_scale = 0.0      # Zero variation = Absolute signal purity
        self.model.noise_scale_w = 0.2    # Minimal duration jitter
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            waveform = outputs.waveform[0].numpy()
            
        # Normalize waveform to prevent clipping before post-processing
        if np.abs(waveform).max() > 0:
            waveform = waveform / np.abs(waveform).max()
            
        return waveform, self.sampling_rate

def get_engine(engine_type="Standard (VITS/IIT Bombay)"):
    """Returns the optimized MMS-VITS engine."""
    print(f"Loading Engine: {engine_type}")
    return MMSVitsEngine()
