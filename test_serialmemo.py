from CogBench.llm_utils.randomllm_serialmemo import RandomSerialMemoryLLM
# from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryExpForLLM
# from CogBench.Experiments.SerialMemoryTask.query_test import SerialMemoryExpForLLM
from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryTaskExpForLLM

# Initialize the new random LLM specialized for Serial Memory
random_serial_llm = RandomSerialMemoryLLM(('random_serial_memory', 100, 0.7))

# Run Serial Memory Experiment
# experiment = SerialMemoryExpForLLM(lambda: random_serial_llm)
experiment = SerialMemoryTaskExpForLLM(random_serial_llm)
df_results = experiment.run_single_experiment(random_serial_llm)

print("\n=== TEST OUTPUT ===")
# print(df_results.head())  # Print first few rows to verify structure
print(df_results)
