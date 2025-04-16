import re
from collections import Counter
import kenlm

model = kenlm.Model('3gram.binary')

def words(text):
    return re.findall(r'\w+', text.lower())

with open('big.txt', 'r', encoding='utf-8') as f:
    WORDS = Counter(words(f.read()))

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N

# === STEP 2: Define QWERTY proximity map ===
QWERTY_ADJACENT_KEYS = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfcx',
    'e': 'wsdr', 'f': 'rtgdvc', 'g': 'tyfhvb', 'h': 'yugjnb',
    'i': 'ujko', 'j': 'uikhmn', 'k': 'ijolm', 'l': 'kop',
    'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
    'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
    'y': 'tghu', 'z': 'asx'
}

# === STEP 3: Define edits with QWERTY bias ===
def edits1_qwerty(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in QWERTY_ADJACENT_KEYS.get(R[0], '')]
    inserts = [L + c + R for L, R in splits if R for c in QWERTY_ADJACENT_KEYS.get(R[0], '')]
    
    return set(deletes + transposes + replaces + inserts)

# === STEP 4: Filter known words ===
def known(words):
    return set(w for w in words if w in WORDS)

# === STEP 5: Suggest next 3 words based on the full sentence ===
def suggest_next_words(input_text, num_suggestions=3):
    # Generate next word candidates by considering common words after the input text
    next_word_candidates = []
    
    # Score each candidate word based on the full sentence
    for word in WORDS:
        sequence = f"{input_text} {word}"
        score = model.score(sequence, bos=False, eos=False)
        next_word_candidates.append((word, score))
    
    # Sort the candidates by score (higher score = better match)
    next_word_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top 'num_suggestions' (top 3 words)
    return next_word_candidates[:num_suggestions]

# === Test it ===
if __name__ == "__main__":
    while True:
        input_text = input("\nEnter a string (or 'q' to quit): ")
        if input_text == 'q': break
        top_3_suggestions = suggest_next_words(input_text)
        print("Top 3 next-word suggestions:")
        for word, score in top_3_suggestions:
            print(f"Word: {word}, Score: {score}")

