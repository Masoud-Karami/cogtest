# Update script logic to read from the JSON synonym file instead of a CSV

import os
import re
import sys
import json
import random
import pandas as pd
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from openai import OpenAI
from CogBench.llm_utils.llms import get_llm
from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryTaskExpForLLM, prepare_study_list

from CogBench.Experiments.SerialMemoryTask.query import prepare_study_list, run_serial_memory_trial


# Setup
load_dotenv("CogBench/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize experiment
experiment = SerialMemoryTaskExpForLLM(get_llm)
JSON_PATH = "CogBench/Experiments/SerialMemoryTask/Dataset/WikiText100_w_with_fallbacks.json"
clean_list, noisy_list = prepare_study_list(experiment, JSON_PATH)
list_lengths = experiment.list_lengths[0]

# Load and prepare the list

run_serial_memory_trial(experiment, clean_list, noisy_list)
print("\n---------------- NOISY STUDY LIST SENT---------------\n")
for i, word in enumerate(noisy_list):
    print(f"{i+1:02d}. {word}")

# Instruction
instruction = experiment.construct_prompt(
    Q_="",
    study_list=noisy_list,
    condition="constant",
    noise=experiment.add_noise
)

raw_output = run_serial_memory_trial(experiment, clean_list, noisy_list)
print("\n--------------- RAW OUTPUT ---------------\n")
print(raw_output)

# Extract recalled words
try:
    parsed = json.loads(raw_output)
    recalled_words = parsed.get("recalled_words", [])
except json.JSONDecodeError:
    print("Warning: JSON parse failed, attempting fallback...")
    match = re.findall(r'"recalled_words"\s*:\s*\[(.*?)\]', output, re.DOTALL)
    recalled_words = re.findall(r'"(.*?)"', match[0]) if match else []

print("\n------------------ STUDY vs RECALL -------------------")
correct = 0
for i, (target, guess) in enumerate(zip(clean_list, recalled_words)):
    target_clean = target.strip()
    guess_clean = guess.strip()

    # Case 1: Distractor + <<silent>>
    if target_clean.startswith("[DISTRACTOR]") and guess_clean.lower() == "<<silent>>":
        mark = "TRUE Recalled! (Silent on Distractor)"
        correct += 1

    # Case 2: Clean word matched
    elif not target_clean.startswith("[DISTRACTOR]") and target_clean.lower() == guess_clean.lower():
        mark = "TRUE Recalled!"
        correct += 1

    # Else: wrong or hallucinated
    else:
        mark = "FALSE Recalled!------"

    print(f"{i+1:02d}. {target:<20} | {guess:<20} {mark}")


# Compare recall to clean study list
# print("\n------------------ STUDY vs RECALL -------------------")
# correct = 0
# for i, (target, guess) in enumerate(zip(clean_study_list, recalled_words)):
#     mark = "TRUE Recalled!" if target.lower(
#     ) == guess.lower() else "FALSE Recalled!------"
#     if mark == "TRUE Recalled!":
#         correct += 1
#     print(f"{i+1:02d}. {target:<{list_lengths}} | {guess:<{list_lengths}} {mark}")

# print(f"\nTotal Correct: {correct}/{len(clean_study_list)}")


# # Prepare scoring
# scorer = StoringSerialMemoryScores()

# columns = [
#     'engine', 'run',
#     'performance_score1', 'performance_score1_name',
#     'performance_score2', 'performance_score2_name',
#     'behaviour_score1', 'behaviour_score1_name',
#     'behaviour_score2', 'behaviour_score2_name',
#     'behaviour_score3', 'behaviour_score3_name',
#     'behaviour_score4', 'behaviour_score4_name'
# ]
# # df_scores = pd.DataFrame(columns=columns)

# # Build results DataFrame
# df_results = pd.DataFrame([{
#     'session': 0,
#     'condition': 'constant',
#     'list_length': len(study_list),
#     'list_index': 0,
#     'trial': 0,
#     'study_list': ','.join(study_list),
#     'recalled_list': ','.join(words),
#     'rel_correct': 0,
#     'init_correct': int(words[0].lower() == study_list[0].lower()) if words else 0,
#     'last_correct': int(words[-1].lower() == study_list[-1].lower()) if words else 0,
#     'forget_rate': None,
#     'ttc_achieved': False,
#     'correct_recall': sum(w1.lower() == w2.lower() for w1, w2 in zip(study_list, words)),
#     'engine': 'gpt-3.5-turbo',
#     'run': 0
# }])

# scored = scorer.get_scores(
#     df_results, df_scores[columns], engine='gpt-3.5-turbo', run=0
# )

# Clean row if shape mismatch
# if len(scored.columns) < len(scored.iloc[-1]):
#     scored.iloc[-1, :] = scored.iloc[-1, :len(scored.columns)]

# print("\n--------------- SERIAL MEMORY METRICS ---------------")
# print(scored.tail(1).T)
