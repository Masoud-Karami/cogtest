import os
import re

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import numpy as np
# https://github.com/NirDiamant/Prompt_Engineering/blob/main/all_prompt_engineering_techniques/prompt-optimization-techniques.ipynb

# Define prompt variations
prompt_a = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

prompt_b = PromptTemplate(
    input_variables=["topic"],
    template="Provide a beginner-friendly explanation of {topic}, including key concepts and an example."
)

# Updated function to evaluate response quality


def evaluate_response(response, criteria):
    """Evaluate the quality of a response based on given criteria.

    Args:
        response (str): The generated response.
        criteria (list): List of criteria to evaluate.

    Returns:
        float: The average score across all criteria.
    """
    scores = []
    for criterion in criteria:
        print(f"Evaluating response based on {criterion}...")
        prompt = f"On a scale of 1-10, rate the following response on {criterion}. Start your response with the numeric score:\n\n{response}"
        response = generate_response(prompt)
        # show 50 characters of the response
        # Use regex to find the first number in the response
        score_match = re.search(r'\d+', response)
        if score_match:
            score = int(score_match.group())
            # Ensure score is not greater than 10
            scores.append(min(score, 10))
        else:
            print(
                f"Warning: Could not extract numeric score for {criterion}. Using default score of 5.")
            scores.append(5)  # Default score if no number is found
    return np.mean(scores)


# Perform A/B test
topic = "machine learning"
response_a = generate_response(prompt_a.format(topic=topic))
response_b = generate_response(prompt_b.format(topic=topic))

criteria = ["Encoding Transparency (Clarity + Structure) includig Clearly separates the study phase from the recall phase, and Reflects temporally ordered input (e.g., “First item is…”).", "informativeness", "engagement"]
score_a = evaluate_response(response_a, criteria)
score_b = evaluate_response(response_b, criteria)

print(f"Prompt A score: {score_a:.2f}")
print(f"Prompt B score: {score_b:.2f}")
print(f"Winning prompt: {'A' if score_a > score_b else 'B'}")
