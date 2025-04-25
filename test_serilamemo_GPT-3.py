import os
import sys
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryTaskExpForLLM
from CogBench.Experiments.SerialMemoryTask.store import StoringSerialMemoryScores

# Setup OpenAI key
load_dotenv("CogBench/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Construct prompt using the original logic from query.py
experiment = SerialMemoryTaskExpForLLM(None)
study_list = [
    "Battig", "Bickley", "DOI", "Hermann", "Intersample", "Joelson", "Kucera", "Landauer", "Lorge", "Madigan",
    "Paivio", "Streeter", "Tarka", "Thorndike", "Yuille", "al", "asymptote", "bigram", "emotionality",
    "et", "etal", "pickList", "preprint", "pronunciability", "yorku"
]
prompt = experiment.construct_prompt(
    "Recall the words you studied.", study_list, condition="constant"
)

# Completion using legacy endpoint (OpenAI SDK v0.27.4)
response = openai.Completion.create(
    engine="gpt-3.5-turbo-instruct",
    prompt=prompt,
    temperature=0,
    max_tokens=300,
)

# Parse and align output
output = response.choices[0].text.strip()
print(f"Raw output: {output}")
words = output.replace('\n', ' ').replace(',', ' ').split()
words = words[:len(study_list)]  # truncate for alignment
print(f"Truncated words: {words}")

# Compare and visualize
print("\n---------------------- PROMPT ------------------------\n")
print(prompt)
print("\n---------------- GPT RESPONSE ------------------------\n")
print(output)

print("\n------------------ STUDY vs RECALL -------------------")
correct = 0
for i, (target, guess) in enumerate(zip(study_list, words)):
    mark = "TRUE Recalled!" if target == guess else "FALSE Recalled!------"
    if mark == "TRUE Recalled!":
        correct += 1
    print(f"{i+1:02d}. {target:<15} | {guess:<15} {mark}")

print(f"\nTotal Correct: {correct}/{len(study_list)}")

# Prepare scoring
scorer = StoringSerialMemoryScores()

# Initialize score DataFrame with required columns
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
    'rel_correct': 0,  # Optional: use experiment.relative_order_scoring()
    'init_correct': int(words[0] == study_list[0]),
    'last_correct': int(words[-1] == study_list[-1]),
    'forget_rate': None,
    'ttc_achieved': False,
    'correct_recall': sum([w1 == w2 for w1, w2 in zip(study_list, words)]),
    'engine': 'gpt-3.5',
    'run': 0
}])

# Score and print
# scored = scorer.get_scores(df_results, df_scores, engine='gpt-3.5', run=0)
scored = scorer.get_scores(
    df_results,
    df_scores[columns],  # ensures consistent columns
    engine='gpt-3.5',
    run=0
)

# Clean patch if extra values were inserted into a new row
if len(scored.columns) < len(scored.iloc[-1]):
    scored.iloc[-1, :] = scored.iloc[-1, :len(scored.columns)]

print("\n=============== SERIAL MEMORY METRICS ===============")
print(scored.tail(1).T)
