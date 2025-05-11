from CogBench.llm_utils.llms import get_llm
from CogBench.base_classes import Experiment
from nltk.corpus import wordnet as wn
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import random
import json
import re

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

        lines = [
            "You will now be presented with a list of words to memorize.",
            "Each word is shown one at a time, followed by a simulated delay.",
            "Some words may be followed by distractors or visual noise.",
            "Focus only on the target words and ignore distractors.",
            "Do not respond until the end of the list is marked."
        ]

        for i, word in enumerate(study_list):
            base = f"word [{i+1}]: \"{word}\""
            lines.append(base)

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

            # lines.append("[1 second later]\n")

        lines.append("<<The list is ended!>>")
        return "\n".join(lines)

    def construct_prompt(self, Q_, study_list, condition, noise=False):
        condition_instr = {
            "constant": "Each trial starts from the same word.",
            "spin": "The start word may change per trial."
        }.get(condition, "")

        noise_instr = "\nSome words may include symbols or distractors. Ignore these during recall." if noise else ""

        return (
            f"{Q_}\n\n"
            "This is a serial memory test for evaluating your ability to remember ordered information.\n"
            f"{condition_instr}{noise_instr}\n"
            "Do not respond until the list is marked as completed using `<<The list is ended!>>`.\n"
            "Once that appears, recall the exact order of original study words.\n"
            "You must respond strictly in the following JSON format:\n"
            "{\n    \"recalled_words\": [\"\", \"\", ..., \"\"]\n}"
        )

    def extract_recalled_list(self, llm_answer, list_length, study_list=None):
        tokens = re.findall(r'\b\w+\b', llm_answer.lower())
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


if __name__ == '__main__':
    experiment = SerialMemoryTaskExpForLLM(get_llm)
    args = experiment.parser.parse_args()
    for run_id in range(args.num_sessions):
        experiment.run_id = run_id
        experiment.run()
