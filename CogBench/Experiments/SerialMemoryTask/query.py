# Assuming the env is in serial_memory.py
from serial_memory import SerialMemoryEnvironment
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
    os.path.dirname(os.path.abspath(__file__))))))  # allows importing CogBench


class SerialMemoryTaskExpForLLM(Experiment):
    """
    Serial Memory Task adapted for LLM-subjected experiments.
    Extends the Experiment class and implements the serial memory paradigm with constant and spin conditions.
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

    def run_single_experiment(self, llm):
        """
        Runs a single experiment of the Serial Memory Task for a given LLM.
        """
        Q_, A_ = llm.Q_A

        list_lengths = self.parser.parse_args().list_lengths
        starting_conditions = self.parser.parse_args().starting_conditions
        num_lists_per_condition = self.parser.parse_args().num_lists_per_condition
        num_sessions = self.parser.parse_args().num_sessions
        max_trials_per_length = {l: m for l, m in zip(
            list_lengths, self.parser.parse_args().max_trials)}

        llm.format_answer = ""

        all_data = []
        word_pool = self.get_word_pool()

        for session in range(num_sessions):
            for condition in starting_conditions:
                for list_length in list_lengths:
                    for list_idx in range(num_lists_per_condition):
                        word_list = np.random.choice(
                            word_pool, list_length, replace=False).tolist()

                        env = SerialMemoryEnvironment(
                            word_list, condition, max_trials_per_length[list_length])
                        list_data = self.run_list(llm, env, Q_, A_)
                        list_data['session'] = session
                        list_data['condition'] = condition
                        list_data['list_length'] = list_length
                        all_data.append(list_data)

        # Concatenate and return
        df = pd.concat(all_data, ignore_index=True)
        return df

    def run_list(self, llm, env, Q_, A_):
        """
        Run the learning of a single list with serial recall under constant or spin conditions.
        """
        history = ""
        trial_data = []

        prev_recall = []
        for trial in range(env.max_trials):
            prompt = self.construct_prompt(env, history, Q_)
            llm_answer = llm.generate(prompt)

            recalled_list = self.extract_recalled_list(
                llm_answer, env.list_length)

            correct_recall = sum(
                [1 for a, b in zip(recalled_list, env.current_study_list) if a == b])
            initial_word_correct = 1 if recalled_list[0] == env.current_study_list[0] else 0

            forgetting_rate = self.compute_forgetting_rate(
                prev_recall, recalled_list)
            prev_recall = recalled_list

            trial_data.append({
                'trial': trial,
                'list_index': env.list_index,
                'study_list': ','.join(env.current_study_list),
                'recalled_list': ','.join(recalled_list),
                'correct_recall': correct_recall,
                'initial_word_correct': initial_word_correct,
                'forgetting_rate': forgetting_rate
            })

            env.next_trial()

            if correct_recall == env.list_length:
                break

        return pd.DataFrame(trial_data)

    def construct_prompt(self, env, history, Q_):
        """
        Construct the prompt for the LLM including study list and recall instruction.
        """
        if env.condition == 'constant':
            instruction = "You will study a list of words. Each time, the list starts from the same first word."
        elif env.condition == 'spin':
            instruction = "You will study a list of words. Each time, the list starts from a different word (spun list)."

        study_list_text = " ".join(env.current_study_list)

        return f"{Q_}\n{instruction}\nStudy list: {study_list_text}\n{history}\nPlease recall the list in order."

    def extract_recalled_list(self, llm_answer, list_length):
        """
        Extract the recalled list from the LLM's generated text.
        Expects a comma-separated or space-separated list.
        """
        words = llm_answer.replace('\n', ' ').replace(',', ' ').split()
        return words[:list_length]

    def compute_forgetting_rate(self, prev_recall, current_recall):
        """
        Computes a forgetting rate between two trials.
        The forgetting rate is the proportion of words previously recalled correctly that are now forgotten.
        """
        if not prev_recall:
            return np.nan

        correctly_retained = sum([1 for w1, w2 in zip(
            prev_recall, current_recall) if w1 == w2])
        return 1 - (correctly_retained / len(prev_recall))

    def get_word_pool(self):
        """
        Load or generate the word pool (using Toronto Word Pool as example).
        """
        return [
            "apple", "table", "sun", "dog", "car", "book", "tree", "house", "river", "mountain",
            "chair", "window", "phone", "garden", "ocean", "computer", "pencil", "hat", "lamp", "shoe",
            "candle", "clock", "door", "guitar", "cat", "plane", "road", "bottle", "bird", "flower"
        ]


if __name__ == '__main__':
    experiment = SerialMemoryTaskExpForLLM(get_llm)
    experiment.run()
