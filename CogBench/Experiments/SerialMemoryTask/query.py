from CogBench.llm_utils.llms import get_llm
from CogBench.base_classes import Experiment
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import random
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))  # allows to import CogBench as a package
# allows importing CogBench as a package

# print("PYTHONPATH set to:", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


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
                                 default=[7, 13, 19], help='List lengths to be tested.')
        self.parser.add_argument('--starting_conditions', nargs='+', default=[
                                 'constant', 'spin'], help='Starting position conditions.')
        self.parser.add_argument('--num_lists_per_condition', type=int,
                                 default=3, help='Number of lists per length per condition.')
        self.parser.add_argument(
            '--num_sessions', type=int, default=3, help='Number of test sessions.')
        self.parser.add_argument('--max_trials', nargs='+', type=int,
                                 default=[70, 130, 160], help='Max trials per list length.')
        parser = self.parser.parse_args()
        self.list_lengths = parser.list_lengths
        self.starting_conditions = parser.starting_conditions
        self.num_lists_per_condition = parser.num_lists_per_condition
        self.num_sessions = parser.num_sessions
        self.max_trials_dict = {l: t for l, t in zip(
            self.list_lengths, parser.max_trials)}
        self.engine = 'unknown'
        self.run_id = 0

    def run_single_experiment(self, llm):
        self.engine = llm.engine_name if hasattr(
            llm, 'engine_name') else 'unknown'
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

                            prompt = self.construct_prompt(
                                Q_, study_list, condition)
                            llm_answer = llm.generate(prompt)
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
                                'recalled_list': ','.join(recalled_list),
                                'rel_correct': relative_correct,
                                'init_correct': first_item_correct,
                                'last_correct': last_item_correct,
                                'forget_rate': forgetting_rate,
                                'ttc_achieved': ttc_achieved,
                                'correct_recall': correct_recall,
                                'errors': sum([r != s.lower() for r, s in zip(recalled_list, study_list)]),
                                'engine': self.engine,
                                'run': self.run_id
                            })

                            if relative_correct == list_length - 1 and first_item_correct:
                                break  # terminate if list is perfectly recalled

        return pd.DataFrame(results)

    def construct_prompt(self, Q_, study_list, condition):
        if condition == "constant":
            instruction = (
                "You are participating in a memory experiment that involves learning a sequence of words.\n"
                "On each trial, you will study a list of words presented in a fixed order.\n"
                "Each list always starts with the same word across trials, but your goal is to learn the **entire sequence** in the correct order.\n"
                "Over multiple study-test trials, try to memorize the exact position of each word.\n\n"
                "This task tests your ability to recall learned sequences and maintain order information across repeated exposure.\n\n"
            )
        elif condition == "spin":
            instruction = (
                "You are participating in a memory experiment that involves recalling word sequences.\n"
                "On each trial, you will study a list of words. The list contains the same words each time, but the **starting point changes on every trial** (like a rotation).\n"
                "Your task is to **recall the sequence exactly as it was presented** on the current trial.\n"
                "This task tests your ability to track sequences even when the starting point shifts.\n\n"
            )
        else:
            raise ValueError(
                f"Unknown condition: {condition}. Should be 'spin' or 'constant'")

        sequential_list = '\n'.join(
            [f"{i+1}. {word}" for i, word in enumerate(study_list)]
        )

        prompt = (
            f"{Q_}\n"
            f"{instruction}"
            "Study phase:\n"
            f"{sequential_list}\n\n"
            "Recall phase:\n"
            "Please type all the words **in the exact order** they were shown, separated by spaces.\n"
            "Do not skip or rearrange any words.\n"
            "Your response:"
        )

        return prompt

    # def extract_recalled_list(self, llm_answer, list_length):
    #     words = llm_answer.replace('\n', ' ').replace(',', ' ').split()
    #     return words[:list_length]

    def extract_recalled_list(self, llm_answer, list_length, study_list=None):
        """
        Parses the LLM's response and extracts a cleaned list of recalled words not hallucinated or partial.
        - Filters tokens by length, alphanumeric content.
        - Optionally ensures words are part of the original study list.
        """
        answer = llm_answer.lower().replace('\n', ' ').replace(',', ' ')
        tokens = answer.split()

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

    def get_word_pool(self):
        return [
            "Battig", "Bickley", "DOI", "Hermann", "Intersample", "Joelson", "Kucera", "Landauer", "Lorge", "Madigan",
            "Paivio", "Streeter", "Tarka", "Thorndike", "Yuille", "al", "asymptote", "bigram", "emotionality",
            "et", "etal", "pickList", "preprint", "pronunciability", "yorku"
        ]


if __name__ == '__main__':
    experiment = SerialMemoryTaskExpForLLM(get_llm)
    args = experiment.parser.parse_args()
    for run_id in range(args.num_runs):
        experiment.run_id = run_id
        experiment.run()
