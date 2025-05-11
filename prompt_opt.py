import os
import re
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables from CogBench/.env
load_dotenv(dotenv_path="CogBench/.env")

# Initialize GPT-4 model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Prompts: your original + 10 LLM-generated variants
test_prompts = [
    # Your original version
    """You are now taking part in a structured memory test composed of multiple sessions, each containing several trials.
In each trial, you will be presented with a list of words, one at a time, in a fixed temporal order.
Each list always starts with the same word across all trials.
Each word will be labeled with its serial position, for example: [Item 1]: restrictive

Some items may include non-alphabetic symbols (e.g., #, %, @, *) at the beginning or end. These symbols are irrelevant and should be ignored.
You may also encounter entries marked with [Distractor], such as: [Distractor]: going. These are not part of the core list.

Strict Rule: FROM NOW ON, do not respond or generate any output until you are sent <<The list is ended!>>

After this signal, recall only the core targeted words in the exact order shown. If you forget a word, use an empty string \"\" in its place.

Respond using this format:
{
    "recalled_words": ["", "", "", ...]
}""",

    # Prompt 1: Baseline Instruction
    "You will be presented with a list of words, one at a time, each labeled with its position (e.g., [Item 1], [Item 2], ...). Your task is to memorize the words in the order presented. After the list ends, recall the words in the same order.",

    # Prompt 2: Emphasizing Primacy and Recency
    "Studies have shown that items at the beginning and end of a list are more easily remembered. Keep this in mind as you are presented with the list of words, one at a time, each labeled with its position. After the list ends, recall the words in the same order.",

    # Prompt 3: Introducing Distractors
    "You will be shown a list of words, one at a time, each labeled with its position. Some items may be labeled as [Distractor]; these are not part of the core list and should be ignored. After the list ends, recall only the core words in the order presented.",

    # Prompt 4: Spin-list
    "In each trial, the list of words will start from a different position, but the internal order remains consistent. Each word is labeled with its position. Your task is to memorize the words in the order presented and recall them in the same order after the list ends.",

    # Prompt 5: Emphasizing Working Memory Constraints
    "Human working memory has limitations. You will be presented with a list of words, one at a time, each labeled with its position. Focus on memorizing as many words as you can in the order presented. After the list ends, recall the words in the same order.",

    # Prompt 6: Semantic Categories
    "You will be shown a list of words, one at a time, each labeled with its position. The words belong to different semantic categories. Pay attention to these categories as they may aid in memorization. After the list ends, recall the words in the same order.",

    # Prompt 7: Serial Position Awareness
    "Be aware that items presented at different positions in a list may be remembered differently. As you are shown the list of words, one at a time, each labeled with its position, try to focus equally on all items. After the list ends, recall the words in the same order.",

    # Prompt 8: Visualization Strategy
    "As you are presented with a list of words, one at a time, each labeled with its position, try to create a mental image for each word. Visualization can aid in memorization. After the list ends, recall the words in the same order.",

    # Prompt 9: Repetition Handling
    "Some words in the list may repeat. You will be shown the list of words, one at a time, each labeled with its position. Pay attention to repeated items as they may affect recall. After the list ends, recall the words in the same order.",

    # Prompt 10: Time Constraint
    "You will be presented with a list of words, one at a time, each labeled with its position. Each word will be displayed for a limited time. Focus on memorizing the words quickly. After the list ends, recall the words in the same order."
]

# Scoring criteria
criteria = [
    "Encoding Transparency (Clarity + Structure), including separation of study and recall phase, and temporal ordering",
    "Informativeness",
    "Engagement"
]


def generate_response(prompt_text):
    """Send prompt to GPT-4 for evaluation."""
    return llm.invoke(prompt_text).content


def evaluate_response(response_text, criteria):
    """Rate prompt quality using GPT-4 based on multiple criteria."""
    scores = []
    for criterion in criteria:
        eval_prompt = f"On a scale of 1 to 10, rate the following prompt on '{criterion}'. Start your response with the numeric score only:\n\n{response_text}"
        rating = generate_response(eval_prompt)
        score_match = re.search(r'\d+', rating)
        score = int(score_match.group()) if score_match else 5
        scores.append(min(score, 10))
    return np.mean(scores)


# Evaluate all prompts
results = []
for i, prompt in enumerate(test_prompts):
    print(f"Evaluating Prompt {i}...")
    score = evaluate_response(prompt, criteria)
    results.append((i, score))
    print(f"Score: {score:.2f}")

# Rank and print
results.sort(key=lambda x: x[1], reverse=True)
print("\nPrompt Ranking:")
for rank, (i, score) in enumerate(results, 1):
    print(f"{rank}. Prompt {i} — Score: {score:.2f}")
