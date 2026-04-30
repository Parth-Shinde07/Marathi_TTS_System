import re

marathi_numbers = {
    '0': 'शून्य', '1': 'एक', '2': 'दोन', '3': 'तीन', '4': 'चार', '5': 'पाच',
    '6': 'सहा', '7': 'सात', '8': 'आठ', '9': 'नऊ', '10': 'दहा',
    # Add full mapping up to 99, 100s, 1000s in production
}

abbreviations = {
    r'डॉ\.': 'डॉक्टर',
    r'प्रा\.': 'प्राध्यापक',
    r'सौ\.': 'सौभाग्यवती',
    r'श्री\.': 'श्रीमान',
    r'कि\.मी\.': 'किलोमीटर',
}

def normalize_marathi_text(text):
    # 1. Expand Abbreviations
    for abbr, full in abbreviations.items():
        text = re.sub(abbr, full, text)
    
    # 2. Convert numbers to words (Simple single digit substitution for demo)
    text = ''.join([marathi_numbers.get(char, char) for char in text])
    
    # 3. Clean Unicode noise (ZWJ/ZWNJ)
    text = text.replace('\u200c', '').replace('\u200d', '')
    
    # 4. Fix inverted Unicode short 'i' matra (ि)
    # OCR sometimes outputs 'ि' + 'द' instead of 'द' + 'ि'
    # We use a negative lookbehind to ensure we only swap if it's NOT already preceded by a consonant
    text = re.sub(r'(?<![\u0915-\u0939])(\u093F)([\u0915-\u0939])', r'\2\1', text)
    
    # 5. Fix common glued words from OCR
    glued_words = {
        r'\bआजचादिवस\b': 'आजचा दिवस',
        r'\bमाझनाव\b': 'माझं नाव',
    }
    for wrong, right in glued_words.items():
        text = re.sub(wrong, right, text)
        
    # 6. Punctuation formatting (preserve for prosody)
    text = re.sub(r'\s+([,।?!.])', r'\1', text) # Remove space before punctuation
    text = re.sub(r'([,।?!.])(?=[^\s])', r'\1 ', text)   # Ensure exactly one space after punctuation
    
    return text.strip()

if __name__ == "__main__":
    test_text = "डॉ. बाबासाहेब आंबेडकर यांनी, संविधानाची निर्मिती केली 1947!"
    print("Original:", test_text)
    print("Normalized:", normalize_marathi_text(test_text))
