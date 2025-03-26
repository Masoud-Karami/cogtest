from CogBench.llm_utils.llms import get_llm
from CogBench.base_classes import Experiment
import argparse
import numpy as np
import pandas as pd
import sys
import os
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))  # allows importing CogBench as a package


class SerialMemoryTaskExpForLLM(Experiment):
    """
    Serial Memory Task adapted for LLM-subjected experiments.
    This implementation is self-contained and does not rely on an external environment class.
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
                                 default=[7, 13, 16], help='Max trials per list length.')
        parser = self.parser.parse_args()
        self.list_lengths = parser.list_lengths
        self.starting_conditions = parser.starting_conditions
        self.num_lists_per_condition = parser.num_lists_per_condition
        self.num_sessions = parser.num_sessions
        self.max_trials_dict = {l: t for l, t in zip(
            self.list_lengths, parser.max_trials)}

    def run_single_experiment(self, llm):
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

                            correct_recall = sum(
                                [1 for a, b in zip(recalled_list, study_list) if a == b])
                            initial_word_correct = int(
                                recalled_list[0] == study_list[0]) if recalled_list else 0
                            forgetting_rate = self.compute_forgetting_rate(
                                prev_recall, recalled_list)
                            prev_recall = recalled_list

                            results.append({
                                'session': session,
                                'condition': condition,
                                'list_length': list_length,
                                'list_index': list_idx,
                                'trial': trial,
                                'study_list': ','.join(study_list),
                                'recalled_list': ','.join(recalled_list),
                                'correct_recall': correct_recall,
                                'initial_word_correct': initial_word_correct,
                                'forgetting_rate': forgetting_rate
                            })

                            if correct_recall == list_length:
                                break  # terminate early if perfectly recalled

        return pd.DataFrame(results)

    def construct_prompt(self, Q_, study_list, condition):
        instruction = {
            'constant': "You will study a list of words. Each time, the list starts from the same first word.",
            'spin': "You will study a list of words. Each time, the list starts from a different word (spin list)."
        }[condition]

        return f"{Q_}\n{instruction}\nStudy list: {' '.join(study_list)}\nPlease recall the list in order:"

    def extract_recalled_list(self, llm_answer, list_length):
        words = llm_answer.replace('\n', ' ').replace(',', ' ').split()
        return words[:list_length]

    def compute_forgetting_rate(self, prev_recall, current_recall):
        if not prev_recall:
            return np.nan
        correctly_retained = sum([1 for w1, w2 in zip(
            prev_recall, current_recall) if w1 == w2])
        return 1 - (correctly_retained / len(prev_recall))

    def get_word_pool(self):
        return [
            "apple", "table", "sun", "dog", "car", "book", "tree", "house", "river", "mountain",
            "chair", "window", "phone", "garden", "ocean", "computer", "pencil", "hat", "lamp", "shoe",
            "candle", "clock", "door", "guitar", "cat", "plane", "road", "bottle", "bird", "flower"
        ]


if __name__ == '__main__':
    experiment = SerialMemoryTaskExpForLLM(get_llm)
    experiment.run()
