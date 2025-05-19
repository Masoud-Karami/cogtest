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


def run_benchmark(engine, only_analysis=False, compare_with=None):
    """
    Runs the benchmark for the specified LLM, focusing only on SerialMemoryTask.

    Parameters:
    - engine (str): Name of the LLM engine.
    - only_analysis (bool): If True, skips experiment and score execution.
    - compare_with (List[str] or None): Other models to compare with in analysis.
    """

    experiments_dir = './Experiments'
    analysis_dir = './Analysis'

    # Focus on SerialMemoryTask only
    target_experiment = 'SerialMemoryTask'
    experiment_path = os.path.join(experiments_dir, target_experiment)

    if not only_analysis:
        print(f'Running experiment: {target_experiment}')
        os.chdir(experiment_path)

        subprocess.run(['python3', 'query.py', '--engine', engine])
        subprocess.run(['python3', 'store.py', '--engines', engine])

        os.chdir('../../')  # Return to root

    # Run analysis
    os.chdir(analysis_dir)
    print(f'Behaviour scores:')
    subprocess.run([
        'python3', 'phenotype_comp.py',
        '--models', engine,
        *(compare_with if compare_with else []),
        '--interest', 'behaviour',
        '--store_id', 'full_run',
        '--print_scores',
        '--print_scores_for', 'human', 'random', engine,
        *(compare_with if compare_with else [])
    ])

    print(f'Performance scores:')
    subprocess.run([
        'python3', 'phenotype_comp.py',
        '--models', engine,
        *(compare_with if compare_with else []),
        '--interest', 'performance',
        '--store_id', 'full_run',
        '--print_scores',
        '--print_scores_for', 'human', 'random', engine,
        *(compare_with if compare_with else [])
    ])
    os.chdir('..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the Serial Memory Task benchmark for a given LLM.')
    parser.add_argument('--engine', type=str, required=True,
                        help='The LLM to run the benchmark on (e.g., gpt-3.5-turbo).')
    parser.add_argument('--compare_with', type=str, nargs='+', default=[],
                        help='Optional list of other models to compare with.')
    parser.add_argument('--only_analysis', action='store_true',
                        help='If set, only run the analysis step.')
    args = parser.parse_args()

    run_benchmark(engine=args.engine, only_analysis=args.only_analysis,
                  compare_with=args.compare_with)
