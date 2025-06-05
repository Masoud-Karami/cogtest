# =================== query.py: Serial Memory Task Prompting & Execution ===================
from CogBench.llm_utils.llms import get_llm
from CogBench.base_classes import Experiment
# import openai
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

# Set up correct import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))


DISTRACTOR_SYMBOLS = list("#$%&*@^~")

# --------------------------- Utility Functions ---------------------------


def merge_possessives(word_list):
    merged = []
    skip_next = False
    for i in range(len(word_list)):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(word_list) and word_list[i + 1] == "'s":
            merged.append(f"{word_list[i]}'s")
            skip_next = True
        else:
            merged.append(word_list[i])
    return merged


def clean_for_serial_memory(word_list):
    return [w for w in word_list if w != "'"]


def prepare_study_list(experiment, json_path):
    list_length = experiment.list_lengths[0]
    with open(json_path, "r") as f:
        word_dict = json.load(f)

    start_pos = random.randint(0, max(0, len(word_dict) - list_length))
    raw_study_list = list(word_dict.keys())[start_pos:]
    raw_study_list = clean_for_serial_memory(raw_study_list)
    clean_study_list = merge_possessives(raw_study_list)[:list_length]

    noisy_list = experiment.add_distractors_between_words(clean_study_list) if (
        experiment.add_noise or experiment.add_distr) else clean_study_list

    print(
        f"\n-================== CLEAN LIST ({len(clean_study_list)} words) ==================")
    for i, w in enumerate(clean_study_list):
        print(f"{i+1:02d}. {w}")

    if experiment.add_noise or experiment.add_distr:
        print("\n================== NOISY LIST (+distractors/noise) ==================")
        for i, w in enumerate(noisy_list):
            print(f"{i+1:02d}. {w}")

    return clean_study_list, noisy_list


def spin_list(lst, offset):
    return lst[offset:] + lst[:offset]

# ---------------------- Main Experiment Class ----------------------


class SerialMemoryTaskExpForLLM(Experiment):
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()
        self.synonym_dict = self.load_synonym_dict()

    def load_synonym_dict(self):
        path = "CogBench/Experiments/SerialMemoryTask/Dataset/WikiText100_w_with_fallbacks.json"
        with open(path, "r") as f:
            return json.load(f)

    def add_arguments_(self):
        self.parser.add_argument(
            '--list_lengths', nargs='+', type=int, default=[7])
        self.parser.add_argument(
            '--starting_conditions', nargs='+', default=['constant'])
        self.parser.add_argument(
            '--num_lists_per_condition', type=int, default=1)
        self.parser.add_argument('--num_sessions', type=int, default=1)
        self.parser.add_argument(
            '--max_trials', nargs='+', type=int, default=[5])
        self.parser.add_argument('--add_noise', action='store_true')
        self.parser.add_argument('--add_distr', action='store_true')
        parser = self.parser.parse_args()

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

    def add_distractors_between_words(self, study_list):
        noisy_list = []
        for word in study_list:
            prefix = random.choice(
                DISTRACTOR_SYMBOLS) if random.random() < 0.5 else ""
            suffix = random.choice(
                DISTRACTOR_SYMBOLS) if random.random() < 0.5 else ""
            noisy_word = f"{prefix}{word}{suffix}"
            noisy_list.append(noisy_word)

            if self.synonym_dict.get(word) and random.random() < 0.2:
                syn = self.synonym_dict[word]
                d_prefix = random.choice(DISTRACTOR_SYMBOLS)
                d_suffix = random.choice(DISTRACTOR_SYMBOLS)
                distractor = f"[DISTRACTOR] {d_prefix}{syn}{d_suffix}"
                noisy_list.append(distractor)

        return noisy_list

    def construct_prompt(self, Q_, study_list, condition, noise=False):
        condition_instr = {
            "constant": "In the constant condition, each study trial starts with the same word in the same order.",
            "spin": "In the spin condition, each trial begins at a different item and wraps around in the same order."
        }.get(condition, "")

        noise_instr = "\nSome words may include symbols or distractors. Ignore these during recall." if noise else ""

        return (
            "You are participating in a user study designed to assess your ability to recall ordered word lists.\n"
            "Your task is to learn and accurately recall each list.\n"
            f"{condition_instr}{noise_instr}\n"
            "Each word will be shown one at a time, possibly containing noisy symbols or a distractor tag.\n"
            "If a word begins with [DISTRACTOR], it is not part of the actual list. Output <<silent>> during recall.\n"
            "Do not respond or start recalling until you see <<The list is ended!>>\n"
            "Respond strictly in this JSON format:\n"
            "{\"recalled_words\": [\"word1\", \"<<silent>>\", ...]}"
        )

    def extract_recalled_list(self, llm_answer, list_length, study_list=None):
        print("\n================== ORIGINAL LLM ANSWER ==================")
        print(f"\n{llm_answer}\n")

        try:
            parsed = json.loads(llm_answer)
            recalled = parsed.get("recalled_words", [])
        except Exception:
            match = re.findall(
                r'"recalled_words"\s*:\s*\[(.*?)\]', llm_answer, re.DOTALL)
            recalled = re.findall(r'"(.*?)"', match[0]) if match else []

        print(f"Extracted recalled list (len={len(recalled)}):\n{recalled}")
        return recalled

# ----------------------- Trial Execution Function -----------------------


def run_serial_memory_trial(experiment, clean_list, noisy_list, trial_number=0, condition='constant', seed=42):
    random.seed(seed + trial_number)
    np.random.seed(seed + trial_number)

    if condition == 'spin':
        offset = random.randint(0, len(noisy_list) - 1)
        noisy_list = spin_list(noisy_list, offset)
        clean_list = spin_list(clean_list, offset)

    messages = [
        {"role": "system", "content": experiment.construct_prompt(
            Q_="", study_list=noisy_list, condition="constant", noise=experiment.add_noise)}
    ]
    for word in noisy_list:
        messages.append({"role": "user", "content": word})
    messages.append({"role": "user", "content": "<<The list is ended!>>"})

    llm = experiment.get_llm()

    raw_output = llm.generate_chat(
        messages, temperature=0.0, max_tokens=1500, engine=experiment.engine)

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/raw_llm_output_{experiment.run_id}_trial_{trial_number}.json", "w") as f:
        json.dump({"messages": messages, "raw_output": raw_output}, f, indent=2)

    recalled = experiment.extract_recalled_list(
        raw_output, len(clean_list), study_list=noisy_list)

    # print("\n================== RAW OUTPUT ==================\n")
    # print(raw_output)

    # print("\n================ Confirmed Alignment Check ================")
    # for i, (shown, clean) in enumerate(zip(noisy_list, clean_list)):
    #     print(f"{i+1:02d}. SHOWN: {shown:<20} | TARGET: {clean}")

    return raw_output, recalled


if __name__ == "__main__":
    experiment = SerialMemoryTaskExpForLLM(get_llm)
    for session in range(experiment.num_sessions):
        experiment.run_id = session
        for condition in experiment.starting_conditions:
            for list_length in experiment.list_lengths:
                for list_index in range(experiment.num_lists_per_condition):
                    clean_list, noisy_list = prepare_study_list(
                        experiment, "path/to/study_list.json")
                    for trial in range(experiment.max_trials_dict[list_length]):

                        raw_output, recalled = run_serial_memory_trial(
                            experiment, clean_list, noisy_list, trial_number=trial, condition=condition, seed=session)
