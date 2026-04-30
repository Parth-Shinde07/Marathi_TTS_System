marathi_map = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh',
    'च': 'c', 'छ': 'ch', 'ज': 'j', 'झ': 'jh',
    'ट': 'T', 'ठ': 'Th', 'ड': 'D', 'ढ': 'Dh', 'ण': 'N',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'S', 'स': 's', 'ह': 'h', 'ळ': 'L'
}
matras = {'ा': 'aa', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au'}

def marathi_g2p(word):
    phonemes = []
    i = 0
    while i < len(word):
        char = word[i]
        if char in marathi_map:
            phonemes.append(marathi_map[char])
            # Check for matra or halant
            if i + 1 < len(word):
                next_char = word[i+1]
                if next_char in matras:
                    phonemes.append(matras[next_char])
                    i += 1
                elif next_char == '्': # Halant (killer stroke)
                    i += 1 # skip adding inherent 'a'
                else:
                    # Inherent 'a'
                    phonemes.append('a')
            else:
                # Schwa Deletion Rule: End of word usually drops inherent 'a' in Marathi
                pass
        i += 1
    return " ".join(phonemes)

if __name__ == "__main__":
    test_word = "मराठी"
    print(f"{test_word} -> {marathi_g2p(test_word)}")
