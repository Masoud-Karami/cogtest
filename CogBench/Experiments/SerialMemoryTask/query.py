import openai
import re
import json
import random
import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from dotenv import load_dotenv
from nltk.corpus import wordnet as wn
from CogBench.base_classes import Experiment
from CogBench.llm_utils.llms import get_llm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

DISTRACTOR_POOL = ["Xyzon", "Nope", "Blur", "Obscure",
                   "Fuzz", "Synthet", "Distracto", "Foobar", "Nebula"]
DISTRACTOR_SYMBOLS = list("#$%&*@^~")


class SerialMemoryTaskExpForLLM(Experiment):
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()

    def add_arguments_(self):
        self.parser.add_argument(
            '--list_lengths', nargs='+', type=int, default=[7])
        self.parser.add_argument(
            '--starting_conditions', nargs='+', default=['constant'])
        self.parser.add_argument(
            '--num_lists_per_condition', type=int, default=1)
        self.parser.add_argument('--num_sessions', type=int, default=1)
        self.parser.add_argument(
            '--max_trials', nargs='+', type=int, default=[2])
        self.parser.add_argument('--add_noise', action='store_true')
        self.parser.add_argument('--add_distr', action='store_true')

        parser = self.parser.parse_args(args=[] if not hasattr(
            sys, 'argv') or __name__ != '__main__' else None)
        self.list_lengths = parser.list_lengths
        self.starting_conditions = parser.starting_conditions
        self.num_lists_per_condition = parser.num_lists_per_condition
        self.num_sessions = parser.num_sessions
        self.max_trials_dict = {l: t for l, t in zip(
            self.list_lengths, parser.max_trials)}
        self.add_noise = parser.add_noise
        self.add_distr = parser.add_distr
        self.engine = 'unknown'
        self.run_id = 0

        if len(parser.max_trials) != len(parser.list_lengths):
            raise ValueError(
                "Each list length must have a corresponding max_trials value.")

    def build_synonym_dict(self, words):
        synonym_dict = {}
        for word in words:
            synsets = wn.synsets(word)
            synonyms = {lemma.name().replace('_', ' ') for syn in synsets for lemma in syn.lemmas(
            ) if lemma.name().lower() != word.lower()}
            synonym_dict[word] = random.choice(
                list(synonyms)) if synonyms else None
        return synonym_dict

    def generate_positionally_tagged_study_list(self, study_list):
        distractor_pool = DISTRACTOR_POOL
        distractor_symbols = DISTRACTOR_SYMBOLS
        synonym_dict = self.build_synonym_dict(study_list)

        lines = []
        for i, word in enumerate(study_list):
            lines.append(f"word [{i+1}]: \"{word}\"")

            if self.add_distr:
                related = synonym_dict.get(word)
                if related:
                    lines.append(f"(Similar to: {related})")

            if self.add_noise:
                noise = ''.join(random.choices(
                    distractor_symbols, k=random.randint(1, 3)))
                lines.append(f"Noise: {noise}")
                if random.random() > 0.5:
                    distractor = random.choice(distractor_pool)
                    d_noise = ''.join(random.choices(
                        distractor_symbols, k=random.randint(1, 3)))
                    lines.append(
                        f"[DISTRACTOR] {d_noise}{distractor}{d_noise}")

        lines.append("<<The list is ended!>>")
        return "\n".join(lines)

    def construct_prompt(self, Q_, study_list, condition, noise=False):
        condition_instr = {
            "constant": "In the constant condition, each study trial starts with the same word in the same order.",
            "spin": "In the spin condition, each trial begins at a different item and wraps around in the same order."
        }.get(condition, "")

        noise_instr = "\nSome words may include symbols or distractors. Ignore these during recall." if noise else ""

        return (
            "You are participating in an experiment involving word lists of 7, 13, or 19 nouns. Your task is to learn and accurately recall each list.\n"
            "Study and test trials alternate across two conditions: constant and spin starting positions.\n"
            f"{condition_instr}\n"
            "Recall the list in the order presented during the most recent study trial.\n"
            "The experiment uses a within-subjects design with three list lengths and both starting conditions.\n"
            "You will complete four sessions (practice and test), alternating between the two starting conditions.\n"
            "You have up to 1 minute per test to recall the words and must indicate completion by saying \"done.\"\n"
            "Your goal is to recall the entire list in order within a limited number of trials.\n"
            "\n"
            "You will now be presented with a list of words to memorize.\n"
            "Each word appears one at a time, possibly followed by noise or a distractor.\n"
            "Focus on study words only. Ignore all other content.\n"
            "Do not respond until the list ends with <<The list is ended!>>.\n"
            "After that, recall the words in exact order.\n"
            "Respond strictly in this JSON format:\n"
            "{\n    \"recalled_words\": [\"\", \"\", ..., \"\"]\n}"
        )

    def add_distractors_between_words(self, study_list):
        distractor_pool = DISTRACTOR_POOL
        distractor_symbols = DISTRACTOR_SYMBOLS

        study_words_lower = {w.lower() for w in study_list}
        filtered_distractors = [
            w for w in distractor_pool if w.lower() not in study_words_lower]
        if not filtered_distractors:
            raise ValueError(
                "Distractor pool is empty after filtering real study words!")

        noisy_list = []

        for word in study_list:
            add_prefix = random.choice([True, False])
            add_suffix = random.choice([True, False])

            prefix_noise = ''.join(random.choices(
                distractor_symbols, k=random.randint(1, 3))) if add_prefix else ""
            suffix_noise = ''.join(random.choices(
                distractor_symbols, k=random.randint(1, 3))) if add_suffix else ""

            noisy_word = f"{prefix_noise}{word}{suffix_noise}"
            noisy_list.append(noisy_word)

            num_distractors = random.randint(0, 4)
            for _ in range(num_distractors):
                distractor = random.choice(filtered_distractors)
                d_prefix = ''.join(random.choices(distractor_symbols, k=random.randint(
                    1, 3))) if random.choice([True, False]) else ""
                d_suffix = ''.join(random.choices(distractor_symbols, k=random.randint(
                    1, 3))) if random.choice([True, False]) else ""
                noisy_distractor = f"[DISTRACTOR] {d_prefix}{distractor}{d_suffix}"
                noisy_list.append(noisy_distractor)

        return noisy_list

    def extract_recalled_list(self, llm_answer, list_length, study_list=None):
        tokens = re.findall(r'\b\w+\b', llm_answer.lower())
        print(f"Extracted tokens form LLM's answer: {tokens}")
        recalled = []
        valid_set = set(w.lower() for w in (study_list or []))
        for token in tokens:
            recalled.append(token)
            if len(recalled) == list_length:
                break
        while len(recalled) < list_length:
            recalled.append("")
        return recalled

    def relative_order_scoring(self, recalled_list, study_list):
        return sum(
            study_list.index(
                recalled_list[i]) + 1 == study_list.index(recalled_list[i + 1])
            for i in range(len(recalled_list) - 1)
            if recalled_list[i] in study_list and recalled_list[i + 1] in study_list
        )

    def compute_forgetting_rate(self, prev_recall, current_recall):
        if not prev_recall:
            return np.nan
        return 1 - sum(1 for a, b in zip(prev_recall, current_recall) if a == b) / len(prev_recall)


def generate_serial_memory_prompt(experiment, json_path, list_size=15):
    """
    Generates a full serial memory prompt for a given experiment instance.

    Args:
        experiment: Instance of SerialMemoryTaskExpForLLM.
        json_path: Path to the JSON file containing the word pool.
        list_size: Number of study words to use.

    Returns:
        A formatted prompt string.
    """
    with open(json_path, "r") as f:
        word_dict = json.load(f)

    study_list = list(word_dict.keys())[:list_size]
    study_list_with_noise = experiment.add_distractors_between_words(
        study_list) if experiment.add_noise else study_list

    print("=== DEBUG: Noisy study list ===")
    for i, word in enumerate(study_list_with_noise):
        print(f'{i+1:03d}. "{word}"')

    # Note: This was previously incorrectly using the clean version regardless of noise
    study_section = experiment.generate_positionally_tagged_study_list(
        study_list_with_noise)

    instructions = experiment.construct_prompt(
        Q_="",
        study_list=study_list_with_noise,
        condition="constant",
        noise=experiment.add_noise
    )

    prompt = f"{instructions}\n\n{study_section}"
    return prompt


if __name__ == '__main__':
    load_dotenv("CogBench/.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    experiment = SerialMemoryTaskExpForLLM(get_llm)
    JSON_PATH = "CogBench/Experiments/SerialMemoryTask/Dataset/WikiText100_w_with_fallbacks.json"

    final_prompt = generate_serial_memory_prompt(
        experiment, JSON_PATH)
    print("\n================ FINAL PROMPT ================\n")
    print(final_prompt)
