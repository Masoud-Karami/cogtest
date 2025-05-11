
# Modified version of 100wordpool.py to save 100 sampled words and their synonyms to JSON
import os
import difflib
import re
import random
import json
from datasets import load_dataset
from nltk.corpus import wordnet as wn
import nltk

# Define output file paths
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(DATASET_DIR, "WikiText100_w_with_syns.json")
FALLBACK_JSON = os.path.join(DATASET_DIR, "WikiText100_w_with_fallbacks.json")


def get_random_100_rows_from_wikitext():
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    total_rows = len(dataset)
    start_idx = random.randint(0, total_rows - 100)
    selected_rows = dataset.select(range(start_idx, start_idx + 100))
    return selected_rows['text']


def clean_and_tokenize(text_blocks):
    full_text = " ".join(text_blocks)
    cleaned = re.sub(r"[^\w\s']", ' ', full_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.split()


def sample_100_consecutive_words(word_list):
    if len(word_list) < 100:
        raise ValueError("Not enough words to sample 100 consecutive ones.")
    start_idx = random.randint(0, len(word_list) - 100)
    return word_list[start_idx: start_idx + 100]


def find_synonym(word):
    synsets = wn.synsets(word)
    synonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            lemma_name = lemma.name().replace('_', ' ')
            if lemma_name.lower() != word.lower():
                synonyms.add(lemma_name)
    return random.choice(list(synonyms)) if synonyms else None


def save_words_to_json(words):
    word_map = {word: find_synonym(word) for word in words}
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(word_map, f, indent=2)
    print(f"Saved 100 words with synonyms to {OUTPUT_JSON}")
    return word_map


def nulltodiffer():
    with open(OUTPUT_JSON, "r") as f:
        synonym_map = json.load(f)

    candidates = list(synonym_map.keys())
    null_targets = [w for w, syn in synonym_map.items() if syn is None]

    for w in null_targets:
        fallback = difflib.get_close_matches(w, candidates, n=1, cutoff=0.7)
        synonym_map[w] = fallback[0] if fallback else None

    with open(FALLBACK_JSON, 'w') as f:
        json.dump(synonym_map, f, indent=2)

    print(f"Replaced nulls and saved to {FALLBACK_JSON}")
    return synonym_map


def main():
    print("Sampling 100 rows from WikiText-103...")
    rows = get_random_100_rows_from_wikitext()

    print("Cleaning and tokenizing...")
    tokens = clean_and_tokenize(rows)

    print("Selecting 100 consecutive words...")
    word_list = sample_100_consecutive_words(tokens)

    print("Saving to JSON...")
    word_map = save_words_to_json(word_list)

    print("Applying lexical fallbacks...")
    updated_map = nulltodiffer()

    print("Done. Sample:")
    print(list(updated_map.items())[:10])


if __name__ == "__main__":
    main()
