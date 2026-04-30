import re

class RuleEngine:
    def __init__(self):
        # We can dynamically load more rules from a JSON dictionary if needed
        self.confusion_matrix = {
            # त vs ट errors (ASR frequently confuses dental and retroflex)
            r'\bताक\b': 'टाक', # 'taak' (throw) often misheard as 'taak' (buttermilk) depending on context,
            # this would ideally be LM checked, but here are safe deterministic ones:
            r'\bगोश्ट\b': 'गोष्ट',
            r'\bकश्ठ\b': 'कष्ट',
            r'\bस्पश्ट\b': 'स्पष्ट',
            
            # Nasal corrections
            r'\bसंबध\b': 'संबंध',
            r'\bआनद\b': 'आनंद',
            r'\bसुदर\b': 'सुंदर',
            
            # Common ASR Hindi-isms
            r'\bमेरा\b': 'माझा',
            r'\bमुझे\b': 'मला',
            r'\bखाना\b': 'जेवण',
            r'\bपानी\b': 'पाणी',
        }

    def fix_nasals(self, text):
        """
        Determines when an anusvara (ं) should be explicitly written as a half-nasal (न्, म्, ण्)
        or vice-versa, depending on standard Marathi conventions (Shuddhalekhan).
        """
        # Modern Marathi strictly prefers Anusvara over half-nasals before certain consonants
        # Example: 'सन्त' -> 'संत'
        text = re.sub(r'न्([तथदध])', r'ं\1', text)
        text = re.sub(r'म्([पफबभ])', r'ं\1', text)
        return text

    def apply_schwa_deletion_fixes(self, text):
        """
        Sometimes ASR outputs an explicit 'a' where it should be silent,
        or halant where a schwa is expected.
        """
        # Remove trailing halants on words where they aren't standard in Marathi
        # (Marathi heavily drops the final schwa naturally, so ASR might insert a halant
        # to represent that phonetic reality, but it's orthographically incorrect).
        text = re.sub(r'्\b', '', text)
        return text

    def handle_english_transliteration(self, text):
        """
        Detects English words mixed in Marathi text and converts them to Devanagari.
        Requires indic-transliteration for advanced usage.
        """
        def is_english(word):
            return bool(re.match(r'^[a-zA-Z]+$', word))

        words = text.split()
        converted = []
        for w in words:
            if is_english(w):
                try:
                    from indic_transliteration import sanscript
                    from indic_transliteration.sanscript import transliterate
                    # Basic fallback ITRANS
                    marathi_w = transliterate(w, sanscript.ITRANS, sanscript.DEVANAGARI)
                    converted.append(marathi_w)
                except ImportError:
                    # If library is missing, keep raw or use a local dict
                    converted.append(w)
            else:
                converted.append(w)
        return " ".join(converted)

    def apply_all_rules(self, text):
        # 1. Phonetic confusions and dictionary mappings
        for wrong, correct in self.confusion_matrix.items():
            text = re.sub(wrong, correct, text)
            
        # 2. Orthographic rules
        text = self.fix_nasals(text)
        text = self.apply_schwa_deletion_fixes(text)
        
        # 3. Transliteration
        text = self.handle_english_transliteration(text)
        
        return text.strip()

if __name__ == "__main__":
    pass
