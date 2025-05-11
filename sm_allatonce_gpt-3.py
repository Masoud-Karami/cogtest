# Update script logic to read from the JSON synonym file instead of a CSV

import os
import re
import sys
import json
import pandas as pd
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from openai import OpenAI
from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryTaskExpForLLM
from CogBench.Experiments.SerialMemoryTask.store import StoringSerialMemoryScores

# Setup OpenAI key
load_dotenv("CogBench/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize experiment
experiment = SerialMemoryTaskExpForLLM(None)

# Load JSON instead of CSV
JSON_PATH = "CogBench/Experiments/SerialMemoryTask/Dataset/WikiText100_w_with_fallbacks.json"

with open(JSON_PATH, "r") as f:
    word_dict = json.load(f)

# Extract just the target words (keys)
study_list = list(word_dict.keys())[:200]

# Step 1: Prepare noisy input
if experiment.add_noise:
    study_list_with_noise = experiment.add_distractors_between_words(
        study_list)
    print("=== DEBUG: Noisy study list ===")
    for i, word in enumerate(study_list_with_noise):
        print(f'{i+1: 03d}. "{word}"')
else:
    study_list_with_noise = study_list

# Step 2: Use *clean* version only for generating prompt memory tag
cleaned_for_tagging = [re.sub(r'^[^\w]*|[^\w]*$', '', w) for w in study_list]

# Step 3: Generate memory seed using clean words only
memory_seed = experiment.generate_positionally_tagged_study_list(
    cleaned_for_tagging)

# Step 4: Construct prompt with noise flag, but show noisy input
prompt = experiment.construct_prompt(
    memory_seed, study_list_with_noise, condition="constant", noise=experiment.add_noise
)

# Send prompt to OpenAI Chat Completion endpoint
client = OpenAI()  # This uses your OPENAI_API_KEY from env
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0,
    max_tokens=512
)

output = response.choices[0].message.content.strip()
print(f"Raw output: {output}")

try:
    parsed = json.loads(output)
    words = parsed.get("recalled_words", [])
except json.JSONDecodeError:
    print("Warning: Failed to parse JSON. Falling back to token splitting.")
    words = output.replace('\n', ' ').replace(',', ' ').split()

# Compare and visualize
print("\n---------------------- PROMPT ------------------------\n")
print(prompt)
print("\n---------------- GPT RESPONSE ------------------------\n")
print(output)

print("\n------------------ STUDY vs RECALL -------------------")
correct = 0
for i, (target, guess) in enumerate(zip(study_list, words)):
    mark = "TRUE Recalled!" if target.lower(
    ) == guess.lower() else "FALSE Recalled!------"
    if mark == "TRUE Recalled!":
        correct += 1
    print(f"{i+1:02d}. {target:<15} | {guess:<15} {mark}")

print(f"\nTotal Correct: {correct}/{len(study_list)}")

# Prepare scoring
scorer = StoringSerialMemoryScores()

columns = [
    'engine', 'run',
    'performance_score1', 'performance_score1_name',
    'performance_score2', 'performance_score2_name',
    'behaviour_score1', 'behaviour_score1_name',
    'behaviour_score2', 'behaviour_score2_name',
    'behaviour_score3', 'behaviour_score3_name',
    'behaviour_score4', 'behaviour_score4_name'
]
df_scores = pd.DataFrame(columns=columns)

# Build results DataFrame
df_results = pd.DataFrame([{
    'session': 0,
    'condition': 'constant',
    'list_length': len(study_list),
    'list_index': 0,
    'trial': 0,
    'study_list': ','.join(study_list),
    'recalled_list': ','.join(words),
    'rel_correct': 0,
    'init_correct': int(words[0].lower() == study_list[0].lower()) if words else 0,
    'last_correct': int(words[-1].lower() == study_list[-1].lower()) if words else 0,
    'forget_rate': None,
    'ttc_achieved': False,
    'correct_recall': sum(w1.lower() == w2.lower() for w1, w2 in zip(study_list, words)),
    'engine': 'gpt-3.5-turbo',
    'run': 0
}])

scored = scorer.get_scores(
    df_results, df_scores[columns], engine='gpt-3.5-turbo', run=0
)

# Clean row if shape mismatch
if len(scored.columns) < len(scored.iloc[-1]):
    scored.iloc[-1, :] = scored.iloc[-1, :len(scored.columns)]

print("\n=============== SERIAL MEMORY METRICS ===============")
print(scored.tail(1).T)
