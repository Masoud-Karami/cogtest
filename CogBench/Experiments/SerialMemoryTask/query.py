from CogBench.llm_utils.llms import get_llm
from CogBench.base_classes import Experiment
from nltk.corpus import wordnet as wn
import nltk
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import random
import pandas as pd
# import ace_tools as tools
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))  # allows to import CogBench as a package
# allows importing CogBench as a package

# print("PYTHONPATH set to:", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
# Define a distractor pool
DISTRACTOR_POOL = ["Xyzon", "Nope", "Blur", "Obscure",
                   "Fuzz", "Synthet", "Distracto", "Foobar", "Nebula"]
DISTRACTOR_SYMBOLS = list("#$%&*@^~")


class SerialMemoryTaskExpForLLM(Experiment):
    """
    Serial Memory Task adapted for LLM-subjected experiments.
    This implementation is self-contained and implements relative order scoring.
    """

    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()

    def add_arguments_(self):
        self.parser.add_argument('--list_lengths', nargs='+', type=int,
                                 default=[7], help='List lengths to be tested.')  # default=[7, 13, 19]
        self.parser.add_argument('--starting_conditions', nargs='+', default=[
                                 'constant'], help='Starting position conditions.')  # default=['constant', 'spin']
        self.parser.add_argument('--num_lists_per_condition', type=int,
                                 default=1, help='Number of lists per length per condition.')  # default=3
        self.parser.add_argument(
            '--num_sessions', type=int, default=1, help='Number of test sessions.')  # default=3
        self.parser.add_argument('--max_trials', nargs='+', type=int,
                                 default=[2], help='Max trials per list length.')  # default=[10, 70, 130, 160]
        self.parser.add_argument('--add_noise', action='store_true',
                                 help='If set, adds noise to the study list during study phase.')
        # self.parser.add_argument(
        #     '--num_runs', type=int, default=1, help='Number of experiment runs.')

        parser = self.parser.parse_args()
        self.list_lengths = parser.list_lengths
        self.starting_conditions = parser.starting_conditions
        self.num_lists_per_condition = parser.num_lists_per_condition
        self.num_sessions = parser.num_sessions
        self.max_trials_dict = {l: t for l, t in zip(
            self.list_lengths, parser.max_trials)}
        self.add_noise = parser.add_noise if parser.add_noise else False
        self.engine = 'unknown'
        self.run_id = 0

        if len(parser.max_trials) != len(parser.list_lengths):
            raise ValueError(
                "Each list length must have a corresponding max_trials value.")

    # Try to build a synonym dictionary for the study words using WordNet
    def build_synonym_dict(words):
        synonym_dict = {}
        for word in words:
            synsets = wn.synsets(word)
            # Gather synonyms excluding the original word
            synonyms = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace('_', ' ')
                    if lemma_name.lower() != word.lower():
                        synonyms.add(lemma_name)
            if synonyms:
                synonym_dict[word] = random.choice(list(synonyms))
            else:
                synonym_dict[word] = None
        return synonym_dict

    # Provided target list (same as before)
    target_words = [
        "Army", "as", "World", "War", "II", "intensified", "He", "first", "went", "for",
        "basic", "training", "to", "Abilene", "Texas", "and", "then", "to", "Brooks", "General",
        "Hospital", "in", "San", "Antonio", "After", "a", "stint", "with", "the", "short",
        "lived", "Army", "Service", "Training", "Program", "Groza", "was", "sent", "with", "the",
        "96th", "Infantry", "Division", "to", "serve", "as", "a", "surgical", "technician", "in"
    ]

    # Generate the synonym dictionary
    synonym_dict = build_synonym_dict(target_words)

    df = pd.DataFrame(list(synonym_dict.items()), columns=[
                      "Target Word", "Random Synonym"])
    df.head()

    def generate_positionally_tagged_study_list(self, study_list):
        """
        Generates a positionally tagged list with optional synonym-based cues
        loaded from synonyms.json.
        """

        # Load precomputed synonym dictionary from disk
        synonym_path = os.path.join(
            os.path.dirname(__file__), "synonyms.json"
        )
        if os.path.exists(synonym_path):
            with open(synonym_path, "r") as f:
                synonym_dict = json.load(f)
        else:
            print("⚠️ Warning: synonym dictionary not found.")
            synonym_dict = {}

        def ordinal_suffix(n):
            if 11 <= n % 100 <= 13:
                return f"{n}th"
            suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
            return f"{n}{suffixes.get(n % 10, 'th')}"

        lines = []
        for i, word in enumerate(study_list):
            related = synonym_dict.get(word, None)

            if related and i == 0:
                line = f"The first starting point, which is related to the {related}, is: {word}"
            elif related:
                line = f"The {ordinal_suffix(i+1)} word is like '{related}' but slightly different — it's: {word}"
            else:
                line = f"The {ordinal_suffix(i+1)} word is: {word}"

            lines.append(line)

        return "The following list is presented:\n" + "\n".join(lines)

    # verify if same ID: reused or different ID: new instance so reinitialized

    def log_llm_instance(self, llm):
        print(f"LLM instance ID: {id(llm)} | Type: {type(llm).__name__}")

    def run_single_experiment(self, llm):
        self.engine = llm.engine_name if hasattr(
            llm, 'engine_name') else 'unknown'
        self.log_llm_instance(llm)
        Q_, A_ = llm.Q_A
        word_pool = self.get_word_pool()
        results = []

        for session in range(self.num_sessions):
            for condition in self.starting_conditions:
                for list_length in self.list_lengths:
                    max_trials = self.max_trials_dict[list_length]

                    for list_idx in range(self.num_lists_per_condition):
                        word_list = random.sample(word_pool, list_length)
                        prev_recall = []

                        for trial in range(max_trials):
                            # Compute spin (rotated) list
                            if condition == "spin":
                                spin_offset = trial % list_length
                                study_list = word_list[spin_offset:] + \
                                    word_list[:spin_offset]
                            else:
                                study_list = word_list
                            # Add noise to the study list if specified
                            # --- Add noise if requested ---
                            if self.add_noise:
                                noisy_study_list = self.add_distractors_between_words(
                                    study_list)
                                # Insert debug output here
                                print("\n--- NOISE DEBUG INFO ---")
                                for original, noisy in zip(study_list, noisy_study_list):
                                    if original != noisy:
                                        print(
                                            f"Original: {original:<15} --> Noisy: {noisy}")

                                print("--- END OF NOISE DEBUG ---\n")
                            else:
                                noisy_study_list = study_list

                            print(f"Trial {trial}: {study_list}")

                            memory_seed = generate_positionally_tagged_study_list(
                                noisy_study_list)
                            prompt = self.construct_prompt(
                                memory_seed, noisy_study_list, condition, noise=self.add_noise)

                            print("\n===================")
                            print("Prompt sent to GPT-3:")
                            print(prompt)
                            print("===================\n")
                            llm_answer = llm.generate(prompt)
                            print("LLM Response:")
                            print(llm_answer)
                            print("===================\n")
                            recalled_list = self.extract_recalled_list(
                                llm_answer, list_length)

                            relative_correct = self.relative_order_scoring(
                                recalled_list, study_list)
                            first_item_correct = int(
                                recalled_list[0] == study_list[0]) if recalled_list else 0
                            last_item_correct = int(
                                recalled_list[-1] == study_list[-1]) if recalled_list else 0
                            forgetting_rate = self.compute_forgetting_rate(
                                prev_recall, recalled_list)
                            prev_recall = recalled_list

                            # Break if list is perfectly recalled
                            correct_recall = sum(
                                [s == r.lower() for s, r in zip(study_list, recalled_list)])
                            ttc_achieved = relative_correct == list_length - 1 and first_item_correct

                            results.append({
                                'session': session,
                                'condition': condition,
                                'list_length': list_length,
                                'list_index': list_idx,
                                'trial': trial,
                                'study_list': ','.join(study_list),
                                'noisy_study_list': ','.join(noisy_study_list),
                                'recalled_list': ','.join(recalled_list),
                                'rel_correct': relative_correct,
                                'init_correct': first_item_correct,
                                'last_correct': last_item_correct,
                                'forget_rate': forgetting_rate,
                                'ttc_achieved': ttc_achieved,
                                'correct_recall': correct_recall,
                                'errors': sum([r != s.lower() for r, s in zip(recalled_list, study_list)]),
                                'engine': self.engine,
                                'run': self.run_id,
                                # <--- new column just added for noisy lists
                                'used_noise': int(self.add_noise)
                            })

                            if relative_correct == list_length - 1 and first_item_correct:
                                break  # terminate if list is perfectly recalled

        return pd.DataFrame(results)

    def construct_prompt(self, Q_, study_list, condition, noise=False):
        """
        Constructs a prompt aligned with the spin-list and serial position encoding principles.
        """
        if condition == "constant":
            instruction = (
                "You were shown a list of words, presented one at a time, starting from the same position each time.\n"
            )
        elif condition == "spin":
            instruction = (
                "You were shown a list of words, presented one at a time."
                "The starting point may have varied across trials, but the order within the list remained consistent.\n"
            )
        else:
            raise ValueError("Invalid condition. Choose 'constant' or 'spin'.")

        if noise:
            instruction += (
                "\nNote: Some words may include extra symbols (e.g., #, $, %, &, *, @, ^, ~) at the beginning or end.\n"
                "These symbols are not part of the actual word. Try to remove them and recall only the core word."
            )

        return (
            f"{Q_}\n\n"
            f"{instruction}\n\n"
            "Your task is to recall the words in the **same order** they appeared."
            "If you do not remember a word at a certain position, leave that slot empty using \"\".\n\n"
            "Please respond in the following JSON format:\n"
            "{\n"
            '    "recalled_words": ["", "", "", ...]  // one item per position\n'
            "}"
            "Recall Phase:\n"
        )

    # def add_distractors_between_words(self, study_list):
    #     """
    #     Adds distractor *words* and optional symbol-only noise to *study* words.
    #     - Inserts 0–4 full distractors between each real word.
    #     - Each real word may also be visually modified with random symbols before/after (not labeled).
    #     """
    #     distractor_pool = DISTRACTOR_POOL
    #     distractor_symbols = DISTRACTOR_SYMBOLS

    #     study_words_lower = {w.lower() for w in study_list}
    #     filtered_distractors = [
    #         w for w in distractor_pool if w.lower() not in study_words_lower
    #     ]
    #     if not filtered_distractors:
    #         raise ValueError(
    #             "Distractor pool is empty after filtering real study words!")

    #     noisy_list = []

    #     for word in study_list:
    #         # --- Add noise to actual study word ---
    #         add_prefix = random.choice([True, False])
    #         add_suffix = random.choice([True, False])

    #         prefix_noise = ''.join(random.choices(
    #             distractor_symbols, k=random.randint(1, 3))) if add_prefix else ""
    #         suffix_noise = ''.join(random.choices(
    #             distractor_symbols, k=random.randint(1, 3))) if add_suffix else ""

    #         noisy_word = f"{prefix_noise}{word}{suffix_noise}"

    #         noisy_list.append(noisy_word)

    #         # --- Add 0–4 full distractors after each study word ---
    #         num_distractors = random.randint(0, 4)
    #         for _ in range(num_distractors):
    #             distractor = random.choice(filtered_distractors)

    #             d_prefix = ''.join(random.choices(distractor_symbols, k=random.randint(
    #                 1, 3))) if random.choice([True, False]) else ""
    #             d_suffix = ''.join(random.choices(distractor_symbols, k=random.randint(
    #                 1, 3))) if random.choice([True, False]) else ""

    #             noisy_distractor = f"[DISTRACTOR] {d_prefix}{distractor}{d_suffix}"
    #             noisy_list.append(noisy_distractor)

    #     return noisy_list

    def extract_recalled_list(self, llm_answer, list_length, study_list=None):
        """
        Parses the LLM's response and extracts a cleaned list of recalled words not hallucinated or partial.
        - Filters tokens by length, alphanumeric content.
        - Optionally ensures words are part of the original study list.
        """
        answer = llm_answer.lower().replace('\n', ' ').replace(',', ' ')
        tokens = answer.split()

        if study_list is None:
            study_list = []
            print("Warning: No study list provided!")

        # Lowercase study list for matching
        valid_set = set(w.lower() for w in study_list)

        # Remove duplicates and hallucinated words
        recalled = []
        for token in tokens:
            # Keep alphanumeric and hyphen
            word = re.sub(r'[^\w\-]', '', token)
            if word in valid_set:
                recalled.append(word)
            else:
                recalled.append(word)

            if len(recalled) == list_length:
                break

        # Pad if too short (to preserve alignment with list_length)
        while len(recalled) < list_length:
            recalled.append("")  # represents omission

        return recalled

    def relative_order_scoring(self, recalled_list, study_list):
        correct_count = 0
        for i in range(len(recalled_list) - 1):
            if recalled_list[i] in study_list and recalled_list[i + 1] in study_list:
                idx1 = study_list.index(recalled_list[i])
                idx2 = study_list.index(recalled_list[i + 1])
                if idx2 == idx1 + 1:
                    correct_count += 1
        return correct_count

    def compute_forgetting_rate(self, prev_recall, current_recall):
        if not prev_recall:
            return np.nan
        correctly_retained = sum([1 for w1, w2 in zip(
            prev_recall, current_recall) if w1 == w2])
        return 1 - (correctly_retained / len(prev_recall))

    # def get_word_pool(self):
    #     return [
    #         "Battig", "Bickley", "DOI", "Hermann", "Intersample", "Joelson", "Kucera", "Landauer", "Lorge", "Madigan",
    #         "Paivio", "Streeter", "Tarka", "Thorndike", "Yuille", "al", "asymptote", "bigram", "emotionality",
    #         "et", "etal", "pickList", "preprint", "pronunciability", "yorku"
    #     ]


if __name__ == '__main__':
    experiment = SerialMemoryTaskExpForLLM(get_llm)
    args = experiment.parser.parse_args()
    for run_id in range(args.num_runs):
        experiment.run_id = run_id
        experiment.run()
