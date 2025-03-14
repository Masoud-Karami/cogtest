import os
import sys
import time
# import anthropic
# import openai
from ..base_classes import RandomLLM 
from ..base_classes import InteractiveLLM
from dotenv import load_dotenv
# Import scripts for the different LLMs
# from .gpt import GPT3LLM, GPT4LLM
# from .anthropic import AnthropicLLM
# from .google import GoogleLLM
from .hf import HF_API_LLM

# Load environment variables (for API keys if required)
load_dotenv()

def get_llm(engine, temp, max_tokens, with_suffix=False):
    """
    Returns the corresponding LLM object based on the engine name.
    Supports:
      - Random Model (`RandomLLM`)
      - Interactive Mode (`InteractiveLLM`)
      - Hugging Face Models (`HF_API_LLM`)
    """
    
    # Initialize step_back and CoT flags
    step_back, cot = False, False

    # Handle suffix-based modifications
    if engine.endswith('_sb'):
        engine = engine[:-3]  # Remove "_sb"
        max_tokens = 350
        step_back = True
    elif engine.endswith('_cot'):
        engine = engine[:-4]  # Remove "_cot"
        max_tokens = 350
        cot = True

    # Select the correct model
    if engine == "interactive":
        llm = InteractiveLLM('interactive')
    elif engine.startswith("hf") or engine.startswith("llama-2"):
        try:
            llm = HF_API_LLM((engine, max_tokens, temp))
        except Exception as e:
            print(f"Error initializing HF API model ({engine}): {e}")
            llm = None
    else:
        print("No key found, defaulting to RandomLLM.")
        llm = RandomLLM(engine)

    # Ensure a valid model was initialized
    if llm is None:
        raise RuntimeError(f"Failed to initialize LLM: {engine}")

    # Set additional attributes
    llm.temperature = temp
    llm.max_tokens = max_tokens
    llm.step_back = step_back
    llm.cot = cot

    return llm
