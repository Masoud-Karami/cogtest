import os
import pandas as pd

from CogBench.llm_utils.randomllm_serialmemo import RandomSerialMemoryLLM
from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryTaskExpForLLM

# Initialize the new random LLM specialized for Serial Memory
random_serial_llm = RandomSerialMemoryLLM(('random_serial_memory', 100, 0.7))

# Run Serial Memory Experiment
# experiment = SerialMemoryExpForLLM(lambda: random_serial_llm)
experiment = SerialMemoryTaskExpForLLM(lambda: random_serial_llm)
df_results = experiment.run_single_experiment(random_serial_llm)

# Save results to data folder
os.makedirs("data", exist_ok=True)
df_results.to_csv(
    "CogBench/Experiments/SerialMemoryTask/scores_data.csv", index=False)

print("\n=== TEST OUTPUT ===")
print(df_results.head())
