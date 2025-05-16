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

DISTRACTOR_SYMBOLS = list("#$%&*@^~")

# Function to merge possessives in a list of words.Remove standalone apostrophes from study lists. They are not behaviorally meaningful in the context of recall. Their presence introduces tokenization and semantic misalignment. Cognitive realism is better preserved by keeping only recallable words.


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

    start_pos = random.randint(0, 10)
    raw_study_list = list(word_dict.keys())[start_pos:]
    raw_study_list = clean_for_serial_memory(raw_study_list)
    clean_study_list = merge_possessives(raw_study_list)[:list_length]

    noisy_list = experiment.add_distractors_between_words(
        clean_study_list) if (experiment.add_noise or experiment.add_distr) else clean_study_list

    print(
        f"\n-==================CLEAN LIST({len(clean_study_list)} words) ==================")
    for i, w in enumerate(clean_study_list):
        print(f"{i+1:02d}. {w}")

    if experiment.add_noise or experiment.add_distr:
        print("\n==================NOISY LIST(+ distr & noise)==================")
        for i, w in enumerate(noisy_list):
            print(f"{i+1:02d}. {w}")

    return clean_study_list, noisy_list


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
            '--list_lengths', nargs='+', type=int, default=[200])
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

    def build_synonym_dict(self, study_list):
        # Placeholder: in practice, return a dictionary of synonyms
        synonym_dict = {word: f"{word}_syn" for word in study_list}
        return synonym_dict

    def generate_positionally_tagged_study_list(self, study_list):
        distractor_symbols = DISTRACTOR_SYMBOLS
        synonym_dict = self.build_synonym_dict(study_list)

        lines = []
        for i, word in enumerate(study_list):
            trial_lines = [f'word [{i+1}]: "{word}"']

            if self.add_distr:
                related = synonym_dict.get(word)
                if related:
                    trial_lines.append(f"(Similar to: {related})")
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
            "If any input word begins with [DISTRACTOR], it is not part of the actual list. You must output <<silent>> during recall for such entries.\n"
            "Do not respond or start recalling until the list ends with <<The list is ended!>>.\n"
            "When list intoduction ended, recall the words in exact order they were presented.\n"
            "Respond strictly using this JSON format:\n"
            '{\n    \"recalled_words": [\"word1", \"word2", \"<<silent>>", ...]\n}'
        )

        return prompt_init

    def add_distractors_between_words(self, study_list):
        distractor_symbols = DISTRACTOR_SYMBOLS

        noisy_list = []

        for word in study_list:
            # Inject character-level noise to the actual word
            prefix = random.choice(
                distractor_symbols) if random.random() < 0.5 else ""
            suffix = random.choice(
                distractor_symbols) if random.random() < 0.5 else ""
            noisy_word = f"{prefix}{word}{suffix}"
            noisy_list.append(noisy_word)

            # With ~20% probability, insert a distractor using known synonym
            synonym = self.synonym_dict.get(word)
            if synonym and random.random() < 0.2:
                d_prefix = random.choice(distractor_symbols)
                d_suffix = random.choice(distractor_symbols)
                distractor = f"[DISTRACTOR] {d_prefix}{synonym}{d_suffix}"
                noisy_list.append(distractor)

        return noisy_list

    def extract_recalled_list(self, llm_answer, list_length, study_list=None):

        # assert study_list is not None, "Pass the correct study_list used in prompting!"
        print("\n================== ORIGINAL LLM ANSWER ==================")
        print(f"\nOriginal LLM answer:\n{llm_answer}\n")

        # Extract raw content from JSON
        try:
            parsed = json.loads(llm_answer)
            recalled = parsed.get("recalled_words", [])
        except Exception:
            # Fallback to regex
            match = re.findall(
                r'"recalled_words"\s*:\s*\[(.*?)\]', llm_answer, re.DOTALL)
            recalled = re.findall(r'"(.*?)"', match[0]) if match else []

        # Do NOT truncate or overwrite anything. Just return.
        print(f"Extracted recalled list (len={len(recalled)}):\n{recalled}")
        return recalled


def run_serial_memory_trial(experiment, clean_list, noisy_list, seed=42):

    random.seed(seed)
    np.random.seed(seed)

    messages = [
        {"role": "system", "content": experiment.construct_prompt(
            Q_="", study_list=noisy_list, condition="constant", noise=experiment.add_noise)}
    ]
    for word in noisy_list:
        messages.append({"role": "user", "content": word})
    messages.append({"role": "user", "content": "<<The list is ended!>>"})

    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=1500
    )

    raw_output = response.choices[0].message.content.strip()

    # STEP 4: Save raw messages and output
    os.makedirs("logs", exist_ok=True)
    with open("logs/raw_llm_output.json", "w") as f:
        json.dump({
            "messages": messages,
            "raw_output": raw_output
        }, f, indent=2)

    recalled = experiment.extract_recalled_list(raw_output, len(
        clean_list), study_list=noisy_list)

    print("\n================== RAW OUTPUT ==================\n")
    print(raw_output)

    print("\n================Confirmed Alignment Check================")
    for i, (shown, clean) in enumerate(zip(noisy_list, clean_list)):
        print(f"{i+1:02d}. SHOWN: {shown:<20} | TARGET: {clean}")

    return raw_output


# def __init__(self, get_llm):
#     super().__init__(get_llm)
#     self.add_arguments_()
#     experiment = SerialMemoryTaskExpForLLM(get_llm)


"""
You are conducting a serial recall user study with LLM models, such as GPT-3-Turbo and Llama-2-7b, to participate. You carefully translate, transfer, design, and send a prompt structure in Python, inspired by human cognitive studies, to make it clear to them their role. Here are some of the rules you must consider when translating:

0) The test structure will be sent to them once, in one prompt.
1) Then, they will be sent a sequence of words, consisting of the target word mixed with noise and distractor words indicated by one input at a time, each in a separate prompt input.
2) Some target words may be mixed by character-level noise (e.g., symbols like #$%&*@^~) 
3) Some input distractor words are marked explicitly as [distractor].

Send task description only once (as a system prompt).
Send each word/distractor as a separate user prompt.
Do NOT wait for LLM to respond after each word.
Only send <<The list is ended!>> to signal recall begins.
Then get a single LLM response, which is parsed and compared to the clean list.
"""
