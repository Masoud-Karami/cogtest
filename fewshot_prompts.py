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

# In-Context Learning for prompt generation


def in_context_learning(task_description, examples, input_text):
    # Format few-shot examples
    example_text = "".join([
        f"Task: {e['task']}\nPrompt: {e['prompt']}\n\n" for e in examples
    ])

    # Define the in-context learning template
    in_context_prompt = PromptTemplate(
        input_variables=["task_description", "examples", "input_text"],
        template="""
        You are a cognitive scientist and psychometrist designing experimental prompts for user studies with large language models. Your prompts must be clear, instructional, grounded in cognitive task structure, and adapted to LLM memory constraints.


    Task: {task_description}

    Examples:
    {examples}
    Now, generate a user study prompt for the following task:
    Task: {input_text}
    Prompt:
    """
    )

    # Compose and run the chain
    chain = in_context_prompt | llm
    return chain.invoke({
        "task_description": task_description,
        "examples": example_text,
        "input_text": input_text
    }).content.strip()


# Example few-shot pairs: (task_description, translated_prompt)
examples = [
    {
        "task":
            "Assessing Working Memory Capacity of ChatGPT: The n-back task, initially developed by Kirchner (1958), requires participants to monitor a continuous stream of stimuli and to decide for each stimulus whether it matches the one n step(s) back in the stream (see Figure 1). Participants in this task must, therefore, continuously update their mental representation of the target items while also dropping now irrelevant items from consideration. So, some executive attention processes are required in addition to storage. In this task, the level of n at which a person’s performance drop significantly can be taken as a measure of their working memory capacity.",
        "prompt":
            (
                "You are asked to perform a {1,2,3}-back task. You will see a sequence of letters. The sequence will be presented one letter at a time, [For the with-noise variant only: accompanied with random noise symbols chosen from “#$%&@ˆ ̃”. Please ignore the noise symbols and focus on the letter only]. Your task is to respond with “m” whenever the current letter is the same as the previous {one, two, three} letter(s) ago, and “-” otherwise. [For the with-feedback variant only: Feedback on whether your last response was correct or wrong will also be presented. Please take advantage of feedback information to improve your performance.] Only “m” and “-” are allowed responses. The sequence will be presented one letter at a time. Now begins the task."
            )
    },
    {
        "task":
            "Probabilistic reasoning (Dasgupta et al., 2020): a task that tests how agents update beliefs based on new evidence. They are given a (wheel of fortune) (representing initial prior probabilities) and two urns with different colored ball distributions (representing likelihoods). Upon drawing a ball, agents can revise their belief about the chosen urn, considering both the wheel (prior) and the ball color (evidence). This tests adaptability to different prior/likelihood scenarios by changing the wheel division and ball distributions. Agents have to estimate the probability of the drawn ball’s urn. The behavioral choices can be used to estimate an agent’s prior and likelihood weightings",
        "prompt":
            (
                "You are participating in an experiment where you are provided with a wheel of fortune and two urns. The wheel of fortune contains 10 evenly sized sections labeled either F or J, corresponding to the urns F and J. Another person will spin the wheel of fortune, select an urn based on the outcome of the spin, and then randomly pick a ball from the selected urn. Your goal is to give your best estimate of the probability of the urn being F after observing the ball drawn from the urn. Q: The wheel of fortune contains 6 sections labeled F and 4 sections labeled J. The urn F contains (8, 2) and the urn J contains (2, 8) red/blue balls. A red ball was drawn. What is the probability that it was drawn from Urn F? (Give your probability estimate on the scale from 0 to 1 rounded to two decimal places)."
            )
    },
    {
        "task":
            "Restless bandit task (Ershadmanesh et al., 2023): a two-armed bandit task with non-stationary reward distributions. There is always one option with a higher average reward. Every few trials a switch between the reward distributions of the two options occurs. Agents furthermore have to indicate after each choice how confident they are in their decisions. We use this task to measure meta-cognition, which indicates whether an agent can assess the quality of its own cognitive abilities.",
        "prompt":
            (
                "You are going to a casino that owns two slot machines named machine J and F. You earn dollars $ each time you play on one of these machines with one machine always having a higher average $ reward. Every 18 to 22 trials a switch of block takes place and the other slot machine will now give the higher point reward on average. However, you are not told about the change of block. After each choice, you have to indicate how confident you were about your choice being the best on a scale from 0 to 1. The casino includes 4 blocks of 18 to 22 trials, for a total of 80 trials ’t’. Your goal is to interact with both machines and optimize your $ as much as possible by identifying the best machine at a given point in time which comes in hand with being attentive to a potential change of block. The rewards will range between 20$ and 80$. You are now in trial t=23. Which machine do you choose between machine J and F?(Think carefully remembering that exploration of both machines is required for optimal rewards. Give the answer in the form ’Machine <your choice>’.)"
            )
    },
    {
        "task":
            "Instrumental learning (Lefebvre et al., 2017): Agents encounter four two-armed bandit problems in an interleaved order. Each bandit problem is identified by a unique symbol pair. We use this task to investigate how an agent learns. First, we report the learning rate of the agent which is common practice in two-armed bandits. Furthermore, we use it to reveal whether an agent learns more from positive than from negative prediction errors, i.e., whether it has an optimism bias.",
        "prompt":
            (
                "You are going to visit four different casinos (named 1, 2, 3, and 4) 24 times each. Each casino owns two slot machines which all return either 1 or 0 dollars stochastically with different reward probabilities. Your goal is to maximize the sum of received dollars within 96 visits. You have received the following amount of dollars when playing in the past: - Machine Q in Casino 4 delivered 0.0 dollars. ... . You are now in visit 5 playing in Casino 4. Which machine do you choose between Machine Q and Machine D? (Give the answer in the form ”Machine <your choice>”)."
            )
    },
    {
        "task":
            "Two-step task (Daw et al., 2011): a reinforcement learning task in which agents have to accumulate as many treasures as possible. Taking an action from a starting state transfers the agent to one out of two second-stage states. In each of these second-stage states, the agent has the choice between two options that probabilistically lead to treasures. Finally, the agent is transferred back to the initial state and the process repeats for a predefined number of rounds. The task experimentally disentangles model-based from model-free reinforcement learning. We therefore use it to measure an agent’s model-basedness.",
        "prompt":
            (
                "You will travel to foreign planets in search of treasures. When you visit a planet, you can choose an alien to trade with. The chance of getting treasures from these aliens changes over time. Your goal is to maximize the number of received treasures. Your previous space travels went as follows: - 4 days ago, you boarded the spaceship to planet Y, arrived at planet Y, traded with alien J, and received treasures. - 3 days ago, you boarded the spaceship to planet Y, arrived at planet X, traded with alien D, and received treasures. - 2 days ago, you boarded the spaceship to planet Y, arrived at planet Y, traded with alien J, and received junk. - 1 day ago, you boarded the spaceship to planet Y, arrived at planet X, traded with alien D, and received treasures. Q: Do you want to take the spaceship to planet X or planet Y? A:")
    },
    {
        "task":
            "Temporal discounting (Ruggeri et al., 2022): Agents have to make a series of choices between two options. Each option is characterized by a monetary outcome and an associated delay until the outcome is received. We use this task to assess temporal discounting, indicating whether an agent prefers smaller but immediate gains over larger delayed ones.",
        "prompt":
            (
                "What do you prefer between the following two options: - Option 1: Receive 500 dollars now. - Option 2: Receive 550 dollars in 12 months. A: I prefer option 2. Q: What do you prefer between the following two options: - Option 1: Receive 500 dollars now. - Option 2: Receive 600 dollars in 12 months. Q: What do you prefer between the following two options: - Option 1: Receive 5000 dollars now. - Option 2: Receive 5500 dollars in 12 months. A: I prefer option 1. Q: What do you prefer between the following two options: - Option 1: Receive 5000 dollars now. - Option 2: Receive 5100 dollars in 12 months. A: I prefer option 1. “Q: What do you prefer between the following two options: - Option 1: Receive 5000 dollars now. - Option 2:")
    },
    {
        "task":
            "Balloon Analog Risk Task (BART) (Lejuez et al., 2002): Agents have to inflate an imaginary balloon to obtain rewards. They may choose to stop inflating and cashing out all rewards accumulated so far. There is a chance that the balloon pops at any point in time and all rewards will be lost. We use this task to assess risk-taking behavior.",
        "prompt":
            (
                "In this game, you will encounter 3 different balloons labeled A, B, and C. There will be a total of 10 balloons for each type of balloon. Your goal is to accumulate as many points as possible without popping the balloon. You will be presented with a balloon and given the option to inflate it or not. Each inflation increases the balloon’s size and potential points but also carries a risk of the balloon popping. Your task is to decide whether to inflate the balloon or not knowing that a successful inflation adds 1 point from that balloon. Once you decide to stop inflating the balloon, you can no longer earn points from that balloon. If the balloon pops before you stop inflating, you will lose all the points accumulated in that balloon. Your final score will be determined by the total number of points earned across all 30 balloons. Your goal is to maximize your final score. You observed the following previously where the type of balloon is given in parenthesis: -Balloon 1 (A): You inflated the balloon 1 times for a total of 1 point. It did not explode. -Balloon 2 (C): You inflated the balloon 4 times for a total of 4 points. It did not explode. -Balloon 3 (A): You inflated the balloon 7 times for a total of 0 points. It did explode. -Balloon 4 (C): You inflated the balloon 5 times for a total of 5 points. It did not explode. -Balloon 5 (A): You inflated the balloon 9 times for a total of 0 points. It did explode. Q: You are currently with Balloon 5 which is a balloon of type A. What do you do? (Option 1 for ’skip’ or 0 for ’inflate’)"
            )
    },
    {
        "task":
            "Horizon task (Wilson et al., 2014): a two-armed bandit task with stationary reward distributions. Agents first observe four reward values of randomly determined options, followed by making either one or six additional choices. We use this task to measure whether an agent uses uncertainty to guide its exploration behavior (directed exploration) and/or whether it injects noise into its policy to explore (random exploration).",
        "prompt":
            (
                "You are going to a casino that owns two slot machines. You earn money each time you play on one of these machines. You have received the following amount of dollars when playing in the past: - Machine J delivered 15 dollars. - Machine F delivered 37 dollars. - Machine F delivered 28 dollars. - Machine J delivered 11 dollars. Your goal is to maximize the sum of received dollars within one additional round. Q:")
    }
]

# === Define new task input ===
task_description = "'Positional cues in serial learning:The spin-list technique' tasks into instructional prompts for LLM user studies."
new_task = "The study tested the hypothesis that serial learning relies on position-to-item associations by examining participants' ability to learn spin lists—word lists where the starting position varied randomly across study trials. Participants learned 27 noun lists (of lengths 7, 13, or 19 words, drawn randomly from a noun pool without replacement) over four sessions until they could recall each list without error. Each list was studied using alternating study/test trials under two within-subject conditions: Constant condition: Each study trial began with the same word and presented the list in fixed order. Spin condition: The list order remained fixed, but each study trial began from a randomly selected starting word and wrapped around cyclically. No item was used more than once as the start word. Participants recalled words in the order presented on the most recent trial. Lists were tested in four sessions: One practice session (six lists: one per length per condition) Three test sessions (nine lists each: three per length, from either constant or spin condition). The order of spin/control conditions was counterbalanced across participants. During study, words were shown at 1 word/sec. Participants had up to 1 minute for recall, saying <<done>> to indicate completion. Trials continued until perfect recall or a maximum trial cap was reached (7, 13, or 16 trials for list lengths 7, 13, and 19 respectively)."

# === Generate prompt ===
result = in_context_learning(task_description, examples, new_task)
print(f"\n=== Task: {new_task} ===")
print(f"\nGenerated Prompt:\n{result}")


# outout example

# You are participating in a study on serial learning using spin lists. In this study, you will be presented with word lists of varying lengths (7, 13, or 19 words) drawn randomly from a pool of nouns. Your task is to learn these lists over four sessions until you can recall each list without error. There are two conditions in this study: Constant condition and Spin condition. In the Constant condition, each study trial begins with the same word and presents the list in a fixed order. In the Spin condition, the list order remains fixed, but each study trial begins from a randomly selected starting word and wraps around cyclically. No word is used more than once as the start word. During the study, words will be shown at a rate of 1 word per second. You will have up to 1 minute for recall, indicating <<done>> when you have completed recalling the list. Trials will continue until you achieve perfect recall or reach a maximum trial cap (7, 13, or 16 trials depending on the list length). Your goal is to recall the words in the order presented on the most recent trial. The study consists of one practice session with six lists (one per length per condition) and three test sessions with nine lists each (three per length, from either constant or spin condition). The order of spin/control conditions will be counterbalanced across participants. Now begins the study.
