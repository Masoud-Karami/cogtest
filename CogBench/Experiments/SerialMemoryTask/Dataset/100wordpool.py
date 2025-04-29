import os
import re
import random
import pandas as pd
from datasets import load_dataset

# Define output directory and file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "selected_100_words.csv")


def get_random_100_rows_from_wikitext():
    """
    Load 100 consecutive rows from WikiText-103 starting at a random point.
    """
    # Load the full dataset index
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    total_rows = len(dataset)
    assert total_rows > 100, "Dataset too small."

    # Randomly pick a start row
    start_idx = random.randint(0, total_rows - 100)
    selected_rows = dataset.select(range(start_idx, start_idx + 100))

    return selected_rows['text']


def clean_and_tokenize(text_blocks):
    """
    Concatenate list of strings, clean, and tokenize into words.
    """
    full_text = " ".join(text_blocks)
    cleaned = re.sub(r"[^\w\s']", ' ', full_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.split()


def sample_100_consecutive_words(word_list):
    """
    Select 100 consecutive words from the full list.
    """
    if len(word_list) < 100:
        raise ValueError("Not enough words to sample 100 consecutive ones.")

    start_idx = random.randint(0, len(word_list) - 100)
    return word_list[start_idx: start_idx + 100]


def save_words_to_csv(words):
    """
    Save list of words to a CSV file, one word per row.
    """
    df = pd.DataFrame({'word': words})
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved 100 words to {OUTPUT_FILE}")


def main():
    print("Sampling 100 rows from WikiText-103...")
    rows = get_random_100_rows_from_wikitext()

    print("Cleaning and tokenizing...")
    tokens = clean_and_tokenize(rows)

    print("Selecting 100 consecutive words...")
    word_list = sample_100_consecutive_words(tokens)

    print("Saving to file...")
    save_words_to_csv(word_list)

    print("Done. Sample:")
    print(word_list[:10])


if __name__ == "__main__":
    main()
