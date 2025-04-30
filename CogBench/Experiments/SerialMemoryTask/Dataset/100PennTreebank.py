import os
import random
import nltk
from nltk.corpus import treebank

# Ensure corpus is available
nltk.download('treebank')

# Constants
NUM_WORDS = 100

# Get current directory (script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "treebank_100_words.txt")

# Load full word list from Penn Treebank
words = treebank.words()

# Check length
if len(words) < NUM_WORDS:
    raise ValueError("Penn Treebank corpus is too small to extract 100 words.")

# Choose a random starting point
start_idx = random.randint(0, len(words) - NUM_WORDS)

# Extract 100 consecutive words
selected_words = words[start_idx: start_idx + NUM_WORDS]

# Lowercase for consistency
selected_words = [word.lower() for word in selected_words]

# Save to .txt file (one word per line)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for word in selected_words:
        f.write(f"{word}\n")

print(f"Saved 100 Penn Treebank words to: {OUTPUT_FILE}")
