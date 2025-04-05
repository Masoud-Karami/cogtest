import os
import sys
from dotenv import load_dotenv
import openai
from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryTaskExpForLLM

# Setup
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
words = output.replace('\n', ' ').replace(',', ' ').split()
words = words[:len(study_list)]  # truncate for alignment

# Compare and visualize
print("\n---------------------- PROMPT ------------------------\n")
print(prompt)
print("\n---------------- GPT RESPONSE ------------------------\n")
print(output)

print("\n------------------ STUDY vs RECALL -------------------")
correct = 0
for i, (target, guess) in enumerate(zip(study_list, words)):
    mark = "✅" if target == guess else "❌"
    if mark == "✅":
        correct += 1
    print(f"{i+1:02d}. {target:<15} | {guess:<15} {mark}")

print(f"\nTotal Correct: {correct}/{len(study_list)}")
