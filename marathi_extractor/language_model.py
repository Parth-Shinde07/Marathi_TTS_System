import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import re

class LanguageModelCorrector:
    def __init__(self, use_mps=True):
        self.device = "mps" if use_mps and torch.backends.mps.is_available() else "cpu"
        # Using a lightweight generative/causal model for perplexity scoring
        # Alternatively, we could use l3cube-pune/marathi-bert-v2 (Masked LM)
        try:
            model_name = "l3cube-pune/marathi-gpt"
            print(f"Loading Marathi LM ({model_name}) on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        except Exception as e:
            print(f"Warning: Could not load Transformer LM. {e}")
            self.model = None

    def calculate_perplexity(self, text):
        """Calculates how 'natural' or grammatically correct a Marathi sentence is."""
        if not self.model or not text.strip():
            return 1.0 # Neutral fallback
            
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            # Cross Entropy Loss
            loss = outputs.loss
            
        # Lower perplexity = better grammar
        perplexity = math.exp(loss.item())
        
        # Convert to a 0-1 confidence score (heuristically)
        # Assuming PPL of 10 is excellent, PPL of 100+ is bad
        score = max(0.0, min(1.0, 1.0 - (perplexity - 10) / 100))
        return score

    def fix_subject_verb_agreement(self, text):
        """
        Rule-based Context-Aware fixes for common ASR grammar mistakes.
        Example: "तो बाजार जाते" (Wrong) -> "तो बाजारात जातो" (Correct)
        """
        # Rule 1: Male singular subject (तो) should end in "तो" (for simple present)
        # ASR often outputs "तो ... जाते" (female ending)
        text = re.sub(r'\b(तो)\b(.*?)\b(जाते)\b', r'\1\2जातो', text)
        text = re.sub(r'\b(तो)\b(.*?)\b(करते)\b', r'\1\2करतो', text)
        
        # Rule 2: Female singular subject (ती) should end in "ते"
        text = re.sub(r'\b(ती)\b(.*?)\b(जातो)\b', r'\1\2जाते', text)
        text = re.sub(r'\b(ती)\b(.*?)\b(करतो)\b', r'\1\2करते', text)
        
        # Rule 3: Neuter singular (ते) should end in "ते" (simple present)
        text = re.sub(r'\b(ते)\b(.*?)\b(जातात)\b', r'\1\2जाते', text) # If singular

        # Rule 4: Locative case marker fixes (बाजार -> बाजारात)
        text = re.sub(r'\bबाजार\s+(जातो|जाते|जातात)\b', r'बाजारात \1', text)
        text = re.sub(r'\bशाळा\s+(जातो|जाते|जातात)\b', r'शाळेत \1', text)

        return text

    def correct(self, text):
        """Pipeline to correct grammar and return confidence."""
        # 1. Apply deterministic context-aware fixes
        corrected_text = self.fix_subject_verb_agreement(text)
        
        # 2. Get neural confidence score
        lm_score = self.calculate_perplexity(corrected_text)
        
        return {
            "corrected_text": corrected_text,
            "lm_score": lm_score
        }

if __name__ == "__main__":
    pass
