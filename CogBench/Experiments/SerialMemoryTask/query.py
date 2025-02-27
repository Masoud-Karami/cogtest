import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) # Allows to import CogBench as a package
from CogBench.base_classes import Experiment
from CogBench.llm_utils.llms import get_llm

class SerialMemoryExpForLLM(Experiment):
    """
    This class represents a Serial Memory Task adapted for LLM-based experiments.
    It extends the base Experiment class and simulates human learning of serial lists under two conditions:
    
    - **Constant Condition**: The same order is used in every trial.
    - **Spin Condition**: A random word is used as the starting point in each trial, wrapping around the list.
    
    The experiment tracks learning progress, recall accuracy, and intertrial forgetting.
    """
    def __init__(self, get_llm):
        super().__init__(get_llm)
        self.add_arguments_()
    
    def add_arguments_(self):
        self.parser.add_argument('--num_runs', type=int, default=10, help='Number of runs')
        self.parser.add_argument('--list_lengths', nargs='+', type=int, default=[7, 13, 19], help='List lengths to be used')
        self.parser.add_argument('--conditions', nargs='+', default=['constant', 'spin'], help='Learning conditions: constant or spin')
        self.parser.add_argument('--max_trials', type=dict, default={7: 7, 13: 13, 19: 16}, help='Max trials per list length')
        self.parser.add_argument('--word_pool', type=str, default='toronto_word_pool.txt', help='Path to the word pool file')
    
    def load_word_pool(self):
        """Loads a set of words from the Toronto Word Pool file."""
        with open(self.parser.parse_args().word_pool, 'r') as f:
            words = [line.strip() for line in f.readlines()]
        return words
    
    def generate_word_list(self, length, word_pool):
        """Generates a unique random word list of a given length."""
        return random.sample(word_pool, length)
    
    def run_single_experiment(self, llm):
        """Runs a single LLM-based Serial Memory Experiment."""
        args = self.parser.parse_args()
        word_pool = self.load_word_pool()
        data = []
        
        for condition in args.conditions:
            for length in args.list_lengths:
                for _ in range(args.num_runs):
                    word_list = self.generate_word_list(length, word_pool)
                    learned = False
                    trial = 0
                    
                    while not learned and trial < args.max_trials[length]:
                        trial += 1
                        
                        if condition == 'constant':
                            study_list = word_list[:]
                        elif condition == 'spin':
                            start_index = random.randint(0, length - 1)
                            study_list = word_list[start_index:] + word_list[:start_index]
                        
                        Q_ = "Study the following list carefully and recall it in order."
                        prompt = f"{Q_}\n{' '.join(study_list)}\nRecall the list in order:"
                        llm.format_answer = "Answer: "
                        recall_response = llm.generate(prompt).strip()
                        recalled_list = recall_response.split()
                        
                        accuracy = sum([1 for i, word in enumerate(recalled_list) if i < length and word == word_list[i]]) / length
                        learned = accuracy == 1.0
                        
                        data.append({
                            'condition': condition,
                            'list_length': length,
                            'trial': trial,
                            'accuracy': accuracy,
                            'learned': learned,
                            'original_list': word_list,
                            'study_list': study_list,
                            'recalled_list': recalled_list
                        })
        
        df = pd.DataFrame(data)
        return df

if __name__ == '__main__':
    experiment = SerialMemoryExpForLLM(get_llm)
    experiment.run()
