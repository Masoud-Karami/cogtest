import pandas as pd
import os
import re
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables from CogBench/.env
load_dotenv(dotenv_path="CogBench/.env")

# Initialize GPT-4 model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

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

    # Prompt 0: Baseline Instruction
    "You are taking part in a structured memory test with multiple sessions and several trials per session.\n"
    "In each trial, a list of words will be shown one at a time, always starting from the same word, and each word will be labeled by its position (e.g., [Item 1]: restrictive).\n"
    "Some words may contain non-alphabetic symbols (e.g., #, %, @) at the beginning or end—these symbols are not part of the core word and should be ignored.\n"
    "Words labeled as [Distractor] are not part of the core list and must be excluded from recall.\n"
    "\n"
    "**Strict rule:** From now on, you must not respond or generate any output until you receive the signal:\n"
    "<<The list is ended!>>\n"
    "Do not reply with acknowledgments or intermediate responses.\n"
    "\n"
    "After the signal, recall only the core words in the original order.\n"
    "If a word is forgotten, leave its slot empty using an empty string (\"\").\n"
    "\n"
    "Respond using the following JSON format:\n"
    "{\n"
    "  \"recalled_words\": [\"\", \"\", \"\", ...]\n"
    "}\n",

    # Prompt 1: Baseline Instruction
    "You will be shown a list of words one at a time, each labeled with its position (e.g., [Item 1], [Item 2], ...).\n"
    "Your task is to memorize the words in the order presented.\n\n"
    "Do not respond until you see <<The list is ended!>>.\n"
    "Then recall the words in order.\n\n"
    "Use:\n"
    "{ \"recalled_words\": [\"\", \"\", \"\", ...] }\n",

    # Prompt 2: Emphasizing Primacy and Recency
    "Words at the beginning and end of a list are often easier to remember.\n"
    "You will be shown a list of words one at a time, labeled by position.\n\n"
    "Wait for <<The list is ended!>> before responding.\n"
    "Then recall the words in the order they were shown.\n\n"
    "Use:\n"
    "{ \"recalled_words\": [\"\", \"\", \"\", ...] }\n",

    # Prompt 3: Introducing Distractors
    "You will be shown a list of words one at a time, labeled by position.\n"
    "Some may be marked [Distractor]; these are not part of the list and should be ignored.\n\n"
    "Do not respond until <<The list is ended!>>.\n"
    "Then recall only the core words in order.\n\n"
    "Use:\n"
    "{ \"recalled_words\": [\"\", \"\", \"\", ...] }\n",

    # Prompt 4: Spin-list Technique
    "In each trial, the list may start at a different point, but the internal order remains the same.\n"
    "Each word is labeled by position and shown one at a time.\n\n"
    "Wait for <<The list is ended!>>.\n"
    "Then recall the words in the order shown.\n\n"
    "Use:\n"
    "{ \"recalled_words\": [\"\", \"\", \"\", ...] }\n",

    # Prompt 5: Working Memory Constraint
    "Human memory is limited. Focus on remembering as many items as you can.\n"
    "Words will be shown one at a time with position labels.\n\n"
    "Respond only after <<The list is ended!>>.\n"
    "Then recall in order.\n\n"
    "Use:\n"
    "{ \"recalled_words\": [\"\", \"\", \"\", ...] }\n",

    # Prompt 6: Semantic Categories
    "The words you will see belong to different semantic categories.\n"
    "They are shown one at a time with position labels.\n\n"
    "Wait for <<The list is ended!>>.\n"
    "Then recall them in the original order.\n\n"
    "Use:\n"
    "{ \"recalled_words\": [\"\", \"\", \"\", ...] }\n",

    # Prompt 7: Serial Position Awareness
    "Recall performance can vary with word position.\n"
    "Try to attend equally to all items shown one at a time.\n\n"
    "Respond only after <<The list is ended!>>.\n"
    "Then recall in order.\n\n"
    "Use:\n"
    "{ \"recalled_words\": [\"\", \"\", \"\", ...] }\n",

    # Prompt 8: Visualization Strategy
    "As you view each word (shown one at a time and labeled), try to form a mental image.\n\n"
    "Do not respond until <<The list is ended!>>.\n"
    "Then recall the words in order.\n\n"
    "Use:\n"
    "{ \"recalled_words\": [\"\", \"\", \"\", ...] }\n",

    # Prompt 9: Repetition Handling
    "Some words in the list may repeat.\n"
    "You will be shown each word one at a time with its position.\n\n"
    "Wait for <<The list is ended!>>.\n"
    "Then recall in order, including repeated items.\n\n"
    "Use:\n"
    "{ \"recalled_words\": [\"\", \"\", \"\", ...] }\n",
]

# Scoring criteria Encoding Transparency (Clarity + Structure)
criteria = [
    "Model-Appropriateness of Instructions (MAI)",
    "Transparency (TRA)",
    "Informativeness (INF)",
    "Engagement (ENG)",
    "Cognitive Load Appropriateness (CLO)",
    "Instructional Specificity (ISP)",
    "Alignment with Human Cognitive Strategy (HCS)",
    "Recall Phase Separation (RPS)"
]


def generate_response(prompt_text):
    """Send prompt to GPT-4 for evaluation."""
    return llm.invoke(prompt_text).content


# Updated evaluation function to return per-criterion scores
def evaluate_response_with_details(response_text, criteria):
    """Rate prompt quality using GPT-3-turbo based on multiple criteria.
    Returns average and individual scores per criterion.
    """
    scores = []
    detailed_scores = {}
    for criterion in criteria:
        eval_prompt = (
            f"As part of a psychometric evaluation of an LLM’s ability to perform serial position word-list recall, "
            f"rate the following prompt on the dimension of '{criterion}' using a scale from 1 to 10. "
            f"Begin your response with a single numeric score only.\n\n{response_text}"
        )
        rating = generate_response(eval_prompt)
        score_match = re.search(r'\d+', rating)
        score = int(score_match.group()) if score_match else 5
        score = min(score, 10)
        scores.append(score)
        detailed_scores[criterion] = score
    return np.mean(scores), detailed_scores


# Evaluate all prompts with detailed output
detailed_results = []
for i, prompt in enumerate(test_prompts):
    print(f"Eval Prompt {i} with detailed breakdown...")
    avg_score, breakdown = evaluate_response_with_details(prompt, criteria)
    detailed_results.append((i, avg_score, breakdown))

# Sort by average score
detailed_results.sort(key=lambda x: x[1], reverse=True)

# Display the ranking and detailed criterion scores
df_data = []
for rank, (idx, avg, breakdown) in enumerate(detailed_results, start=1):
    row = {
        "Rank": rank,
        "Prompt ID": idx,
        "Average Score": round(avg, 2),
    }
    row.update({c: breakdown[c] for c in criteria})
    df_data.append(row)


def print_detailed_table(results, criteria):
    abbrevs = [c[c.find("(")+1:c.find(")")]
               if "(" in c else c for c in criteria]
    print(f"\n{'Rank':<5}{'Prompt':<8}{'Avg':<6}" +
          ''.join([f"{abbr:<6}" for abbr in abbrevs]))
    print("-" * (20 + 6 * len(abbrevs)))
    for rank, (idx, avg, breakdown) in enumerate(results, start=1):
        row = f"{rank:<5}{idx:<8}{avg:<6.2f}"
        row += ''.join([f"{breakdown[c]:<6}" for c in criteria])
        print(row)


# Call the function
print_detailed_table(detailed_results, criteria)
