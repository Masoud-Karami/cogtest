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
from CogBench.Experiments.SerialMemoryTask.store import StoringSerialMemoryScores
from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryTaskExpForLLM

# Setup
load_dotenv("CogBench/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize experiment
experiment = SerialMemoryTaskExpForLLM(get_llm)
JSON_PATH = "CogBench/Experiments/SerialMemoryTask/Dataset/WikiText100_w_with_fallbacks.json"
list_lengths = experiment.list_lengths[0]
# Load and prepare the list


def prepare_study_list(experiment, json_path):
    list_length = experiment.list_lengths[0]
    with open(json_path, "r") as f:
        word_dict = json.load(f)

    start_pos = random.randint(0, 10)
    clean_study_list = list(word_dict.keys())[
        start_pos:start_pos + list_length]
    noisy_list = experiment.add_distractors_between_words(
        clean_study_list) if experiment.add_noise else clean_study_list
    return clean_study_list, noisy_list


clean_study_list, noisy_study_list = prepare_study_list(experiment, JSON_PATH)
print("\n================== NOISY STUDY LIST SENT ==================\n")
for i, word in enumerate(noisy_study_list):
    print(f"{i+1:02d}. {word}")

# Instruction
instruction = experiment.construct_prompt(
    Q_="",
    study_list=noisy_study_list,
    condition="constant",
    noise=experiment.add_noise
)

# Construct multi-turn message list
messages = [{"role": "system", "content": instruction}]
for word in noisy_study_list:
    messages.append({"role": "user", "content": word})

# Signal end of list
messages.append({"role": "user", "content": "<<The list is ended!>>"})

# Call OpenAI API
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0,
    max_tokens=1500
)

# Extract and parse recall
output = response.choices[0].message.content.strip()
print("\n================== RAW OUTPUT ==================\n")
print(output)

# Extract recalled words
try:
    parsed = json.loads(output)
    recalled_words = parsed.get("recalled_words", [])
except json.JSONDecodeError:
    print("Warning: JSON parse failed, attempting fallback...")
    match = re.findall(r'"recalled_words"\s*:\s*\[(.*?)\]', output, re.DOTALL)
    recalled_words = re.findall(r'"(.*?)"', match[0]) if match else []

# Compare recall to clean study list
print("\n------------------ STUDY vs RECALL -------------------")
correct = 0
for i, (target, guess) in enumerate(zip(clean_study_list, recalled_words)):
    mark = "TRUE Recalled!" if target.lower(
    ) == guess.lower() else "FALSE Recalled!------"
    if mark == "TRUE Recalled!":
        correct += 1
    print(f"{i+1:02d}. {target:<{list_lengths}} | {guess:<{list_lengths}} {mark}")

print(f"\nTotal Correct: {correct}/{len(clean_study_list)}")


# # Load API key
# load_dotenv("CogBench/.env")
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize experiment
# experiment = SerialMemoryTaskExpForLLM(None)

# # Path to the JSON word list
# JSON_PATH = "CogBench/Experiments/SerialMemoryTask/Dataset/WikiText100_w_with_fallbacks.json"

# list_lengths = experiment.list_lengths[0]
# # Generate full memory task prompt
# # prompt = generate_serial_memory_prompt(experiment, JSON_PATH, list_size)
# prompt, study_list_with_noise, clean_study_list = generate_serial_memory_prompt(
#     experiment, JSON_PATH)

# # Send prompt to OpenAI
# client = OpenAI()
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": prompt}
#     ],
#     temperature=0,
#     max_tokens=2500
# )

# output = response.choices[0].message.content.strip()
# print("\n================== RAW OUTPUT ==================")
# print(f"Raw output: {output}")

# # Try parsing JSON output
# try:
#     parsed = json.loads(output)
#     asitis_recalled = parsed.get("recalled_words", [])
# except json.JSONDecodeError:
#     print("Warning: Failed to parse JSON. Trying partial fix.")
#     match = re.findall(
#         r'"asitis recalled words"\s*:\s*\[(.*?)\]', output, re.DOTALL)
#     if match:
#         asitis_recalled = re.findall(r'"(.*?)"', match[0])
#     else:
#         asitis_recalled = []

# # Compare recall with target study list
# print("\n------------------ STUDY vs RECALL -------------------")
# correct = 0
# for i, (target, guess) in enumerate(zip(clean_study_list, asitis_recalled)):
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

# print("\n=============== SERIAL MEMORY METRICS ===============")
# print(scored.tail(1).T)
