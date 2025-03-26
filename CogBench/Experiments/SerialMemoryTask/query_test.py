from CogBench.llm_utils.llms import get_llm
from CogBench.base_classes import Experiment
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))  # allows importing CogBench as a package


class SerialMemoryTaskExpForLLM(Experiment):
    """
    This class represents the Serial Memory experiment adapted for LLMs.
    The LLM is tested on learning word lists with either constant or randomized (spin) start positions.
    """

    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()

    def add_arguments_(self):
        self.parser.add_argument('--list_lengths', nargs='+', type=int,
                                 default=[7, 13, 19], help='List lengths to test')
        self.parser.add_argument('--start_position', type=str, default='constant',
                                 choices=['constant', 'spin'], help='Condition type')
        self.parser.add_argument(
            '--num_runs', type=int, default=100, help='Number of LLM runs per condition')
        self.parser.add_argument(
            '--max_trials', type=int, default=16, help='Maximum trials per list')
        args = self.parser.parse_args()
        self.list_lengths = args.list_lengths
        self.start_position = args.start_position
        self.num_runs = args.num_runs
        self.max_trials = args.max_trials

    def run_single_experiment(self, llm):
        """Runs an LLM on a single serial memory experiment.
        Args:
            llm (LLM): LLM object used for word generation and recall.
        Returns:
            df (pd.DataFrame): Dataframe with experiment results.
        """
        Q_, A_ = llm.Q_A
        # llm.random_fct = self.random_fct
        def llm_generate(x): return llm.generate(
            x).split()  # Generate words in list format

        word_pool = ["apple", "banana", "cherry", "dog", "elephant", "fish", "grape", "house", "island", "jungle",
                     "kangaroo", "lion", "mountain", "notebook", "octopus", "penguin", "queen", "river", "sunflower"]  # Example words

        data = []

        for list_length in self.list_lengths:
            for _ in range(self.num_runs):
                # Randomly sample unique words
                word_list = random.sample(word_pool, list_length)
                trial = 0
                learned = False
                recall_history = []
                initial_prompt = f"Memorize this list of words in order: {' '.join(word_list)}."

                while trial < self.max_trials and not learned:
                    # Determine start position (Constant: 0, Spin: Random)
                    start_pos = 0 if self.start_position == "constant" else random.randint(
                        0, list_length - 1)

                    # Prompt the LLM for recall
                    recall_prompt = initial_prompt + \
                        f"\nStart recall from word {start_pos+1}:"
                    recall_attempt = llm_generate(recall_prompt)

                    # Compute correctness using relative order scoring
                    correct_count = self.relative_order_scoring(
                        recall_attempt, word_list[start_pos:])
                    perfect_recall = correct_count == len(
                        word_list[start_pos:])

                    # Store results
                    data.append([list_length, self.start_position, trial +
                                1, start_pos, correct_count, perfect_recall])
                    recall_history.append((trial, recall_attempt))

                    if perfect_recall:
                        learned = True
                    trial += 1

        df = pd.DataFrame(data, columns=[
                          'list_length', 'condition', 'trial', 'start_position', 'correct_count', 'perfect_recall'])
        return df

    def relative_order_scoring(self, recall_attempt, target_list):
        """Scores recall based on relative order, ignoring strict positional accuracy.
        Args:
            recall_attempt (list): Words recalled by the LLM.
            target_list (list): The correct word sequence.
        Returns:
            int: Number of words correctly recalled in relative order.
        """
        correct_count = 0
        for i in range(len(recall_attempt) - 1):
            if recall_attempt[i] in target_list and recall_attempt[i+1] in target_list:
                if target_list.index(recall_attempt[i+1]) == target_list.index(recall_attempt[i]) + 1:
                    correct_count += 1
        return correct_count


if __name__ == '__main__':
    experiment = SerialMemoryTaskExpForLLM(get_llm)
    experiment.run()
