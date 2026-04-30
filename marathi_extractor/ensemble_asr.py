import whisper
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import difflib

class EnsembleASR:
    def __init__(self, use_mps=True):
        self.device = "mps" if use_mps and torch.backends.mps.is_available() else "cpu"
        print(f"Loading Ensemble ASR on {self.device}...")
        
        # 1. Primary: Whisper (Strong grammar, prone to hallucination)
        self.whisper_model = whisper.load_model("large-v3", device=self.device)
        
        # 2. Secondary: IndicWav2Vec2 (Strictly phonetic, low hallucination)
        w2v2_model_id = "ai4bharat/indicwav2vec_v1_marathi"
        try:
            self.w2v2_processor = Wav2Vec2Processor.from_pretrained(w2v2_model_id)
            self.w2v2_model = Wav2Vec2ForCTC.from_pretrained(w2v2_model_id).to(self.device)
        except Exception as e:
            print(f"Warning: Could not load Wav2Vec2 model. Fallback to Whisper-only mode. {e}")
            self.w2v2_model = None

    def transcribe_whisper(self, audio_path):
        result = self.whisper_model.transcribe(audio_path, language="mr")
        return result["text"].strip()

    def transcribe_w2v2(self, audio_path):
        if self.w2v2_model is None:
            return ""
            
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            
        # Mono channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        inputs = self.w2v2_processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.w2v2_model(input_values).logits
            
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.w2v2_processor.batch_decode(predicted_ids)[0]
        return transcription.strip()

    def get_agreement_score(self, text1, text2):
        """Returns a similarity score between 0.0 and 1.0"""
        if not text1 or not text2:
            return 0.5 # Neutral if one fails
        
        # Using difflib for character-level SequenceMatcher
        # In a strict production setup, use word-level WER (Word Error Rate)
        matcher = difflib.SequenceMatcher(None, text1.replace(" ", ""), text2.replace(" ", ""))
        return matcher.ratio()

    def extract(self, audio_path):
        """Runs both models and returns the primary text and the agreement score."""
        whisper_text = self.transcribe_whisper(audio_path)
        w2v2_text = self.transcribe_w2v2(audio_path)
        
        # We always trust Whisper's formatting/punctuation over raw Wav2Vec2,
        # but we use W2V2's output to calculate an 'ASR Agreement Score'
        agreement = self.get_agreement_score(whisper_text, w2v2_text)
        
        return {
            "primary_text": whisper_text,
            "secondary_text": w2v2_text,
            "asr_score": agreement
        }

if __name__ == "__main__":
    # Test script
    pass
