import librosa
import re

class PhoneticAligner:
    def __init__(self, chars_per_second=13.5):
        # Average speaking rate for Marathi (syllables/characters per second)
        self.chars_per_second = chars_per_second
        
    def text_to_phoneme_count(self, text):
        """
        Estimates the phonetic length of the text.
        In Devanagari, we count aksharas (syllable units) rather than raw characters,
        as a consonant+matra often forms one spoken unit.
        """
        if not text:
            return 0
            
        # Remove spaces and punctuation
        clean_text = re.sub(r'[\s।॥!?%,.:;…\.\-\'\"]+', '', text)
        
        # Count dependent vowel signs (matras), halants, and anusvaras as modifiers 
        # that don't add full spoken duration.
        modifiers = len(re.findall(r'[\u0900-\u0903\u093A-\u094D\u0950-\u0957]', clean_text))
        
        raw_chars = len(clean_text)
        
        # Phonetic length is roughly the number of base consonants/vowels
        # (Raw length minus the modifiers which just attach to the base)
        estimated_syllables = raw_chars - (modifiers * 0.5) 
        
        return max(1.0, estimated_syllables)

    def verify_alignment(self, audio_path, text):
        """
        Compares actual audio duration to expected duration based on phonetic count.
        Returns a confidence score [0.0 - 1.0].
        """
        try:
            duration = librosa.get_duration(filename=audio_path)
        except Exception as e:
            print(f"Error loading audio for alignment: {e}")
            return 0.0

        # Hard bounds check
        if duration > 20.0 or duration < 0.5:
            # Extreme outliers are heavily penalized
            return 0.1

        phonetic_length = self.text_to_phoneme_count(text)
        expected_duration = phonetic_length / self.chars_per_second
        
        # Calculate divergence
        difference = abs(duration - expected_duration)
        
        # If the difference is huge (e.g. > 50% of the duration), it's highly likely 
        # ASR hallucinated or truncated the text.
        ratio = difference / duration
        
        if ratio < 0.2:
            return 1.0 # Perfect alignment
        elif ratio < 0.4:
            return 0.8 # Good alignment (speaker might just be slow/fast)
        elif ratio < 0.6:
            return 0.5 # Suspicious
        else:
            return 0.2 # Critical mismatch (Hallucination detected)

if __name__ == "__main__":
    pass
