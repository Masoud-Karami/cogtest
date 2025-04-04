from CogBench.llm_utils.gpt import GPT3LLM  # or GPT3LLM
from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryTaskExpForLLM
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))


# Load API key from .env
load_dotenv("CogBench/.env")
gpt_key = os.getenv("OPENAI_API_KEY")
# Define model info
llm_info = (gpt_key, "text-davinci-003", None)  # Change model as needed

# Initialize LLM
llm = GPT3LLM(llm_info)

# Initialize experiment
experiment = SerialMemoryTaskExpForLLM(None)

# Prepare a test prompt
study_list = [
    "Battig", "Bickley", "DOI", "Hermann", "Intersample", "Joelson", "Kucera", "Landauer", "Lorge", "Madigan",
    "Paivio", "Streeter", "Tarka", "Thorndike", "Yuille", "al", "asymptote", "bigram", "emotionality",
    "et", "etal", "pickList", "preprint", "pronunciability", "yorku"
]
prompt = experiment.construct_prompt(
    "Recall the words you studied.", study_list, "constant")

# Run generation
response = llm.generate(prompt)
print("=== Prompt ===\n", prompt)
print("\n=== GPT Response ===\n", response)
