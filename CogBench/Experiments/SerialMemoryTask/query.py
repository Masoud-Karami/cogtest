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
                            prompt = self.construct_prompt(
                                Q_, noisy_study_list, condition, noise=self.add_noise)
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

    # TODO: Add noisy related prompt instruction!

    # def construct_prompt(self, Q_, study_list, condition):
    #     if condition == "constant":  # tests your ability to recall learned sequences and maintain order information across repeated exposure
    #         instruction = (
    #             "You are participating in a memory experiment that involves learning a sequence of words.\n"
    #             "On each trial, you will study a list of words presented one word at a time, and in a fixed order.\n"
    #             "Each list always starts with the same word across trials, but your goal is to learn the **entire sequence** in the correct order.\n"
    #             "Over multiple study-test trials, try to memorize the exact position of each word.\n\n")
    #     elif condition == "spin":
    #         instruction = (
    #             "You are participating in a memory experiment that involves recalling word sequences.\n"
    #             "On each trial, you will study a list of words. The list contains the same words each time, but the **starting point changes on every trial** (like a rotation).\n"
    #             "Your task is to **recall the sequence exactly as it was presented** on the current trial.\n"
    #             "This task tests your ability to track sequences even when the starting point shifts.\n\n"
    #         )
    #     else:
    #         raise ValueError(
    #             f"Unknown condition: {condition}. Should be 'spin' or 'constant'")

    #     sequential_list = '\n'.join(
    #         [f"{i+1}. {word}" for i, word in enumerate(study_list)]
    #     )

    #     prompt = (
    #         f"{Q_}\n"
    #         f"{instruction}"
    #         "Study phase:\n"
    #         f"{sequential_list}\n\n"
    #         "Recall phase:\n"
    #         "Please type all the words **in the exact order** they were shown, separated by spaces.\n"
    #         "Do not skip or rearrange any words.\n"
    #         "Your response:"
    #     )

    #     return prompt

    def construct_prompt(self, Q_, study_list, condition, noise=False):
        """
        Constructs a prompt for the Serial Memory Task, incorporating advanced prompting techniques.

        Args:
            Q_ (str): Base query or instruction.
            study_list (list of str): List of study words, possibly including distractors.
            condition (str): 'constant' or 'spin' to indicate list presentation condition.
            noise (bool): Flag indicating presence of distractor words.

        Returns:
            str: Formatted prompt string.
        """
        # Base instruction based on condition
        if condition == "constant":
            instruction = (
                "You will study a list of words presented one at the time and in a fixed order.\n"
                "Your task is to memorize and recall the words in the exact order presented.\n"
            )
        elif condition == "spin":
            instruction = (
                "You will study a list of words where the starting point may vary each time, "
                "but the internal order remains consistent. \n"
                "Your task is to memorize and recall the words in the order they are presented.\n"
            )
        else:
            raise ValueError("Invalid condition. Choose 'constant' or 'spin'.")

        # Add distractor instruction if noise is present
        if noise:
            instruction += (
                " Note: Some words will be labeled as [DISTRACTOR]. "
                "Ignore these distractor words and focus only on the main study words."
            )

        # Format the study list with numbering
        study_phase = "\n".join(
            f"{idx + 1}. {word}" for idx, word in enumerate(study_list))

        # Construct the full prompt
        prompt = (
            f"{Q_}\n\n"
            f"{instruction}\n\n"
            "Study Phase:\n"
            f"{study_phase}\n\n"
            "Recall Phase:\n"
            "Please list the study words in the order presented, separated by spaces. "
            "Do not include any distractor words in your response.\n"
            "Your answer:"
        )

        return prompt

    # def extract_recalled_list(self, llm_answer, list_length):
    #     words = llm_answer.replace('\n', ' ').replace(',', ' ').split()
    #     return words[:list_length]

    # def add_noise_to_list(self, study_list):
    #     noisy_list = []
    #     noise_pool = ["Xyzon", "Nope", "Blur",
    #                   "Obscure", "Fuzz", "Synthet", "Noise"]
    #     for word in study_list:
    #         r = random.random()
    #         if r < 0.2:
    #             # Replace with random noise word
    #             noisy_list.append(random.choice(noise_pool))
    #         elif r < 0.4:
    #             # Add special character noise
    #             noisy_word = self.add_noise_to_stimulus(word)
    #             noisy_list.append(noisy_word)
    #         elif r < 0.5:
    #             # Masked word
    #             noisy_list.append("_" * len(word))
    #         else:
    #             noisy_list.append(word)
    #     return noisy_list

    # def add_noise_to_list(self, study_list):
    #     """
    #     Adds distractors or mild noise around study words.
    #     Noise types:
    #     - Inject random noise words (not part of the study set).
    #     - Add symbols at the beginning or end (but not inside words).
    #     - Mask some words with underscores.

    #     Returns:
    #         list: Noisy study list.
    #     """
    #     noisy_list = []
    #     noise_pool = ["Xyzon", "Nope", "Blur",
    #                   "Obscure", "Fuzz", "Synthet", "Noise"]
    #     study_set = set(w.lower() for w in study_list)

    #     for word in study_list:
    #         r = random.random()

    #         if r < 0.2:
    #             # Insert a random distractor (not in original list)
    #             distractor = random.choice(noise_pool)
    #             noisy_list.append(f"[DISTRACTOR] {distractor}")

    #         elif r < 0.4:
    #             # Add special characters at beginning or end only
    #             special_symbols = random.choice(
    #                 ["#", "$", "%", "&", "*", "@", "^", "~"])
    #             if random.random() < 0.5:
    #                 noisy_word = f"{special_symbols}{word}"
    #             else:
    #                 noisy_word = f"{word}{special_symbols}"
    #             noisy_list.append(noisy_word)

    #         elif r < 0.5:
    #             # Mask the word fully with underscores
    #             masked_word = "_" * len(word)
    #             noisy_list.append(masked_word)

    #         else:
    #             # Keep original word untouched
    #             noisy_list.append(word)

    #     return noisy_list

    # def add_noise_to_stimulus(self, stimulus):
    #     """
    #     Adds small noise to a word by injecting random special characters.
    #     """
    #     noise_symbols = list("#$%&*@^~")
    #     noise_length = random.randint(1, 3)  # add 1-3 symbols
    #     noise = ''.join(random.choices(noise_symbols, k=noise_length))
    #     noisy_word = list(stimulus + noise)
    #     random.shuffle(noisy_word)
    #     return ''.join(noisy_word)


def add_distractors_between_words(self, study_list):
    """
    Adds labeled distractors randomly between real study words.
    - Real words are preserved and not modified.
    - Distractors are inserted in between, labeled as [DISTRACTOR] word.
    """
    # Define a distractor pool
    distractor_pool = ["Xyzon", "Nope", "Blur", "Obscure",
                       "Fuzz", "Synthet", "Distracto", "Foobar", "Nebula"]
    distractor_symbols = list("#$%&*@^~")

    # Filter out distractors that accidentally match real words
    study_words_lower = {w.lower() for w in study_list}
    filtered_distractors = [
        w for w in distractor_pool if w.lower() not in study_words_lower]

    if not filtered_distractors:
        raise ValueError(
            "Distractor pool is empty after filtering real study words!")

    noisy_list = []

    for word in study_list:
        # Always add the real word
        noisy_list.append(word)

        # Random chance to insert a distractor after this word
        if random.random() < 0.5:  # 50% chance to insert distractor after each real word
            distractor_word = random.choice(filtered_distractors)

            # Small chance to add noise symbols before/after the distractor
            if random.random() < 0.5:
                add_at_start = random.choice([True, False])
                noise = ''.join(random.choices(
                    distractor_symbols, k=random.randint(1, 3)))
                if add_at_start:
                    distractor_word = noise + distractor_word
                else:
                    distractor_word = distractor_word + noise

            # Label the distractor
            noisy_list.append(f"[DISTRACTOR] {distractor_word}")

    return noisy_list

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
