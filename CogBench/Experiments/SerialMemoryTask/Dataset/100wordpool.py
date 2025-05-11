# run: python 100wordpool.py --num_words 150

import os
import difflib
import re
import random
import json
import argparse
from datasets import load_dataset
from nltk.corpus import wordnet as wn
import nltk

# Define output file paths
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(DATASET_DIR, "WikiText100_w_with_syns.json")
FALLBACK_JSON = os.path.join(DATASET_DIR, "WikiText100_w_with_fallbacks.json")


def get_random_rows_from_wikitext(n_rows):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    total_rows = len(dataset)
    start_idx = random.randint(0, total_rows - n_rows)
    selected_rows = dataset.select(range(start_idx, start_idx + n_rows))
    return selected_rows['text']


def clean_and_tokenize(text_blocks):
    full_text = " ".join(text_blocks)
    cleaned = re.sub(r"[^\w\s']", ' ', full_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.split()


def sample_consecutive_words(word_list, sample_size):
    if len(word_list) < sample_size:
        raise ValueError(
            f"Not enough words to sample {sample_size} consecutive ones.")
    start_idx = random.randint(0, len(word_list) - sample_size)
    return word_list[start_idx: start_idx + sample_size]


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
    print(f"Saved {len(words)} words with synonyms to {OUTPUT_JSON}")
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
    parser = argparse.ArgumentParser(
        description="Sample word list with synonyms from WikiText")
    parser.add_argument("--num_words", type=int, default=100,
                        help="Number of consecutive words to sample")
    args = parser.parse_args()

    print(f"Sampling {args.num_words} rows from WikiText-103...")
    rows = get_random_rows_from_wikitext(args.num_words)

    print("Cleaning and tokenizing...")
    tokens = clean_and_tokenize(rows)

    print(f"Selecting {args.num_words} consecutive words...")
    word_list = sample_consecutive_words(tokens, args.num_words)

    print("Saving to JSON...")
    word_map = save_words_to_json(word_list)

    print("Applying lexical fallbacks...")
    updated_map = nulltodiffer()

    print("Done. Sample:")
    print(list(updated_map.items())[:10])


if __name__ == "__main__":
    main()
