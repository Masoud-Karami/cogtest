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

# Randomly select a starting position for the study list
starting_position = random.randint(0, 10)


class SerialMemoryTaskExpForLLM(Experiment):
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()
        self.add_noise = True
        # self.add_distr = True

    def add_arguments_(self):
        self.parser.add_argument(
            '--list_lengths', nargs='+', type=int, default=[20])
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

    # def build_synonym_dict(self, words):
    #     synonym_dict = {}
    #     for word in words:
    #         synsets = wn.synsets(word)
    #         synonyms = {lemma.name().replace('_', ' ') for syn in synsets for lemma in syn.lemmas(
    #         ) if lemma.name().lower() != word.lower()}
    #         synonym_dict[word] = random.choice(
    #             list(synonyms)) if synonyms else None
    #     return synonym_dict

    def build_synonym_dict(self, study_list):
        # Placeholder: in practice, return a dictionary of synonyms
        synonym_dict = {word: f"{word}_syn" for word in study_list}
        return synonym_dict

    def generate_positionally_tagged_study_list(self, study_list):
        distractor_pool = DISTRACTOR_POOL
        distractor_symbols = DISTRACTOR_SYMBOLS
        synonym_dict = self.build_synonym_dict(study_list)

        lines = []
        for i, word in enumerate(study_list):
            trial_lines = [f'word [{i+1}]: "{word}"']

            if self.add_distr:
                related = synonym_dict.get(word)
                if related:
                    trial_lines.append(f"(Similar to: {related})")

            # if self.add_noise:
            #     noise = ''.join(random.choices(
            #         distractor_symbols, k=1))
            #     trial_lines.append(f"Noise: {noise}")
            #     if random.random() > 0.2:
            #         distractor = random.choice(distractor_pool)
            #         d_noise = ''.join(random.choices(
            #             distractor_symbols, k=random.randint(1, 3)))
            #         trial_lines.append(
            #             f"[DISTRACTOR] {d_noise}{distractor}{d_noise}")

            # Add instruction for silence after each input
            # trial_lines.append("Do not respond. Wait for the next word.")
            lines.append("\n".join(trial_lines))

        lines.append("<<The list is ended!>>")
        return "\n".join(lines)

    def construct_prompt(self, Q_, study_list, condition, noise=False):
        condition_instr = {
            "constant": "In the constant condition, each study trial starts with the same word in the same order.",
            "spin": "In the spin condition, each trial begins at a different item and wraps around in the same order."
        }.get(condition, "")

        noise_instr = "\nSome words may include symbols or distractors. Ignore these during recall." if noise else ""

        prompt_init = (
            "You are participating in a user study designed to assess your ability to recall ordered word lists. Your task is to learn and accurately recall each list.\n"
            "The study follows a within-subjects design with alternating **study** and **test** trials across two conditions:\n"
            f"{condition_instr}\n"
            "Recall the list in the order presented during the most recent study trial.\n"
            "You will complete four total sessions (one practice and three test sessions), alternating between these two conditions.\n"
            "Each word list is studied and tested until it is recalled perfectly or a maximum number of trials is reached.\n"
            "During each test, you will have up to 1 minute to recall the list and must indicate when you are finished by saying “done”.\n"
            "You will be presented with a list of words to memorize.\n"
            "**Each word will be shown one at a time**, possibly containing noisy symbols or followed by a `[DISTRACTOR]` word.\n"
            "Your task is to **focus only on the clean study word** presented in each step. Ignore any distractors or symbol-based noise.\n"
            "If a position originally contained a distractor, you must **output nothing** in that position—use the special marker <<silent>>:\n"
            "Do not respond or start recalling until the list ends with <<The list is ended!>>.\n"
            "After that, recall the words in exact order.\n"
            "Respond strictly in this JSON format:\n"
            "{\n    \"recalled_words\": [\"\", \"\", ..., \"\"]\n}"
        )

        return prompt_init

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
        # tokens = re.findall(r'\b\w+\b', llm_answer.lower())
        # print(f"Extracted tokens form LLM's answer: {tokens}")
        # recalled = []
        # valid_set = set(w.lower() for w in (study_list or []))
        # for token in tokens:
        #     recalled.append(token)
        #     if len(recalled) == list_length:
        #         break
        # while len(recalled) < list_length:
        #     recalled.append("")
        # return recalled

        print(f"Original LLM answer:\n{llm_answer}\n")
        # Match all quoted strings: "word"
        recalled = re.findall(r'"(.*?)"', llm_answer)
        # Preserve original order and content
        recalled = recalled[:list_length]
        # Pad if fewer than list_length
        while len(recalled) < list_length:
            recalled.append("")

        return recalled


def generate_serial_memory_prompt(experiment, json_path):
    """
    Generates a full serial memory prompt for a given experiment instance.

    Args:
        experiment: Instance of SerialMemoryTaskExpForLLM.
        json_path: Path to the JSON file containing the word pool.
        list_size: Number of study words to use.

    Returns:
        A formatted prompt string.
    """
    list_lengths = experiment.list_lengths[0]
    with open(json_path, "r") as f:
        word_dict = json.load(f)

    study_list = list(word_dict.keys())[
        starting_position:list_lengths+starting_position]
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
    return prompt, study_list_with_noise, study_list


if __name__ == '__main__':
    load_dotenv("CogBench/.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    experiment = SerialMemoryTaskExpForLLM(get_llm)
    JSON_PATH = "CogBench/Experiments/SerialMemoryTask/Dataset/WikiText100_w_with_fallbacks.json"

    final_prompt, stlwn = generate_serial_memory_prompt(
        experiment, JSON_PATH)
    print("\n================ FINAL PROMPT ================\n")
    print(final_prompt)
