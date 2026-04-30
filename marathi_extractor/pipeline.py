import json
import os
from .ensemble_asr import EnsembleASR
from .language_model import LanguageModelCorrector
from .phonetic_aligner import PhoneticAligner
from .rule_engine import RuleEngine

class MarathiExtractionPipeline:
    def __init__(self, use_mps=True):
        print("Initializing High-Precision Marathi Extraction Pipeline...")
        self.asr = EnsembleASR(use_mps=use_mps)
        self.lm = LanguageModelCorrector(use_mps=use_mps)
        self.aligner = PhoneticAligner()
        self.rules = RuleEngine()
        
        self.feedback_db_path = "correction_feedback.json"
        self._load_feedback_db()

    def _load_feedback_db(self):
        if os.path.exists(self.feedback_db_path):
            with open(self.feedback_db_path, "r", encoding="utf-8") as f:
                self.feedback_db = json.load(f)
            # Inject learned rules into the rule engine
            for wrong, right in self.feedback_db.items():
                self.rules.confusion_matrix[f"\\b{wrong}\\b"] = right
        else:
            self.feedback_db = {}

    def save_feedback(self, wrong_text, correct_text):
        """Auto-Feedback Loop: Learns from manual corrections"""
        self.feedback_db[wrong_text] = correct_text
        with open(self.feedback_db_path, "w", encoding="utf-8") as f:
            json.dump(self.feedback_db, f, ensure_ascii=False, indent=4)
        print(f"Learned correction: '{wrong_text}' -> '{correct_text}'")

    def process_audio(self, audio_path):
        """
        Executes the 5-stage pipeline for a single audio file.
        Returns: (Clean Text, Confidence Score, Flag)
        """
        
        # STAGE 1: Multi-Model Extraction
        asr_result = self.asr.extract(audio_path)
        base_text = asr_result["primary_text"]
        
        if not base_text:
            return "", 0.0, "FAILED"

        # STAGE 2 & 4: Context/Rules Correction
        # Apply rules first, then LM grammar check
        rule_corrected = self.rules.apply_all_rules(base_text)
        lm_result = self.lm.correct(rule_corrected)
        final_text = lm_result["corrected_text"]
        
        # STAGE 3: Phonetic Alignment Verification
        phonetic_score = self.aligner.verify_alignment(audio_path, final_text)
        
        # STAGE 8: Confidence Scoring System
        # Weights: ASR Agreement (30%), LM Grammar (30%), Phonetic Alignment (40%)
        total_confidence = (
            (asr_result["asr_score"] * 0.30) + 
            (lm_result["lm_score"] * 0.30) + 
            (phonetic_score * 0.40)
        )
        
        # Determine Flag
        if total_confidence >= 0.85:
            flag = "OK"
        elif total_confidence >= 0.60:
            flag = "REVIEW"
        else:
            flag = "REJECT" # Likely complete hallucination or bad audio
            
        return {
            "text": final_text,
            "confidence": round(total_confidence, 3),
            "flag": flag,
            "metrics": {
                "asr_agreement": round(asr_result["asr_score"], 3),
                "lm_grammar": round(lm_result["lm_score"], 3),
                "phonetic_alignment": round(phonetic_score, 3)
            }
        }

if __name__ == "__main__":
    # Test script usage
    # pipeline = MarathiExtractionPipeline()
    # result = pipeline.process_audio("sample.wav")
    # print(f"Audio | {result['text']} | {result['confidence']} | {result['flag']}")
    pass
