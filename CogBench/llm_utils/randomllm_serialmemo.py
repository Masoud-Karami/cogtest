import random
import os
import sys
from CogBench.base_classes import RandomLLM


class RandomSerialMemoryLLM(RandomLLM):
    def __init__(self, llm_info):
        super().__init__(llm_info)

    def generate(self, text, temp=None, max_tokens=None):
        """Overrides the generate function to avoid missing attribute errors."""
        return self.random_fct()

    def random_fct(self):
        """Generates random sequences to simulate Serial Memory recall."""
        # Predefined example word lists (7, 13, 19 words each)
        example_lists = {
            7: ["apple", "banana", "cherry", "dog", "elephant", "fish", "grape"],
            13: ["apple", "banana", "cherry", "dog", "elephant", "fish", "grape", "house", "island",
                 "jungle", "kangaroo", "lion", "mountain"],
            19: ["apple", "banana", "cherry", "dog", "elephant", "fish", "grape", "house", "island", "jungle", "kangaroo", "lion", "mountain", "notebook", "octopus", "penguin", "queen", "river", "sunflower"]
        }

        # Randomly pick a list length
        list_length = random.choice([7, 13, 19])
        word_list = example_lists[list_length]

        # Randomly shuffle and return a partial recall
        # Simulates imperfect recall
        recall_length = random.randint(3, list_length)
        return " ".join(random.sample(word_list, recall_length))
