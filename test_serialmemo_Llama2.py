import os
import pandas as pd
from CogBench.llm_utils.llms import get_llm
from CogBench.Experiments.SerialMemoryTask.query import SerialMemoryTaskExpForLLM


def run_serialmemory_with_llm(engine_name):
    print(f"Running Serial Memory Task for: {engine_name}")
    llama = get_llm(engine=engine_name, temp=0.7, max_tokens=100)
    experiment = SerialMemoryTaskExpForLLM(lambda *args: llama)
    df = experiment.run_single_experiment(llama)

    # Save in the same directory as the experiment
    output_dir = "CogBench/Experiments/SerialMemoryTask"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"scores_data_{engine_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}\n")
    return df


if __name__ == '__main__':
    models = ['llama-2-7b-hf', 'llama-2-7b-chat-hf']
    for model in models:
        run_serialmemory_with_llm(model)
