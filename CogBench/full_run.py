"""
This script is used to run the entire benchmark for a chosen LLM. 
It runs all the experiments, stores the required values, and prints and plots the main metrics.

Usage:
    python3 full_run.py --engine <LLM> [--compare_with <models>]

Arguments:
    --engine: The LLM to run the benchmark on. This is a required argument.
    --compare_with: The models to compare against. This is an optional argument but I am not going to use it to test serialMemoryTask. However, if not provided, it defaults to ['gpt-4', 'claude-2'].
    --only_analysis: If set, only run the analysis and skip the experiment running and storing steps. This is an optional argument.

Functions:
    run_benchmark(engine): Runs the benchmark for the specified LLM.

Note:
    The fitting of scores is generally fast for all experiments, except for the InstrumentalLearning experiment which can be very slow. 
    Please be patient when running this experiment.
    After the analysis, a summary table is printed with the scores for the chosen agent, as well as the human & random agents and the reference scores for the models specified with the `--compare_with` flag. 
    The performance and behavior normalized scores versus the models specified with the `--compare_with` flag are also plotted. 
    The plots are saved in the `./Analysis/plots/phenotypes/full_runs{interest}.pdf` directory.
"""

import os
import argparse
import subprocess
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_benchmark(engine):
    """
    This function runs the benchmark for the specified LLM. It runs all the experiments, stores the required values, 
    and prints and plots the main metrics.

    Arguments:
        engine: The LLM to run the benchmark on.

    Returns:
        None
    """
    experiments_dir = './Experiments'
    analysis_dir = './Analysis'

    # Define folders to exclude
    all_experiments = {'ProbabilisticReasoning', 'HorizonTask', 'RestlessBandit',
                       'InstrumentalLearning', 'TwoStepTask', 'BART', 'SerialMemoryTask', 'TemporalDiscounting'}
    # if exlude all, replace with set() instead of exc_exp{''}
    # excluded_experiments = set()
    excluded_experiments = {'ProbabilisticReasoning', 'HorizonTask', "TwoStepTask",
                            'RestlessBandit', 'InstrumentalLearning', 'BART', 'TemporalDiscounting'}

    # Add folder names you want to skip 'TwoStepTask'
    focusing_folders = list(all_experiments - excluded_experiments)
    if not args.only_analysis:
        # Get all the experiment folders
        experiment_folders = [f.path for f in os.scandir(
            experiments_dir) if f.is_dir()]

        # Filter to only include focusing folders
        experiment_folders = [
            f for f in experiment_folders if os.path.basename(f) in focusing_folders]

        print(f'Focusing tasks: {focusing_folders}')

        for task in experiment_folders:
            folder_name = os.path.basename(task)

            # Run query.py and store.py for each experiment
            os.chdir(task)
            print(f'Running experiment {folder_name}')
            subprocess.run(['python3', 'query.py', '--engines', engine])
            print(
                f'Storing the behavioral scores for experiment {folder_name}')
            subprocess.run(['python3', 'store.py', '--engines', engine])
            os.chdir('../..')  # Go back to the root directory

    # Run phenotype_comp.py in the Analysis folder
    os.chdir(analysis_dir)
    print(f'Behaviour scores:')
    subprocess.run(['python3', 'phenotype_comp.py', '--models', args.engine, '--print_scores', '--print_scores_for', 'human', 'random', args.engine])
    print(f'Performance scores:')
    subprocess.run(['python3', 'phenotype_comp.py', '--models', args.engine, '--print_scores', '--print_scores_for', 'human', 'random', args.engine])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run the entire benchmark for a chosen LLM.')
    parser.add_argument('--engine', type=str, required=True,
                        help='The LLM to run the benchmark on.')
    parser.add_argument('--compare_with', type=str, nargs='*', default=[], help='The models to compare against.')

    parser.add_argument('--only_analysis', action='store_true',
                        help='If set, only run the analysis and skip the experiment running and storing steps.')
    args = parser.parse_args()
    #if args.compare_with is None:
    #	args.compare_with = []
    run_benchmark(args.engine)
