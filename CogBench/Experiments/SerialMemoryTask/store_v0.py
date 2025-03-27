import argparse
import numpy as np
import pandas as pd
import sys
import os
import statsmodels.formula.api as smf
from tqdm import tqdm
from CogBench.base_classes import StoringScores

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))  # Allows importing CogBench as a package


class StoringSerialMemoryScores(StoringScores):
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        pass  # No extra args needed beyond base parser

    def get_all_scores(self):
        args = self.parser.parse_args()
        engines = args.engines

        scores_csv_name = f"scores_data{'V' + args.version_number if args.version_number != '1' else ''}.csv"
        data_folder = f'data{"V"+args.version_number if args.version_number != "1" else ""}'

        if 'all' in engines:
            engines = [os.path.splitext(file)[0]
                       for file in os.listdir(data_folder)]

        storing_df = pd.read_csv(scores_csv_name) if os.path.isfile(
            scores_csv_name) else pd.DataFrame(columns=self.columns)

        for engine in tqdm(engines):
            print(
                f"Scoring for engine: {engine}-------------------------------------------")
            path = f"{data_folder}/{engine}.csv"
            full_df = pd.read_csv(path)
            storing_df = self.get_scores(full_df, storing_df, engine, run=0)
            storing_df.to_csv(scores_csv_name, index=False)

    def get_scores(self, df, storing_df, engine, run):
        data = {
            'correct_recall': df['correct_recall'],
            'initial_word_correct': df['initial_word_correct'],
            'forgetting_rate': df['forgetting_rate']
        }

        df_clean = pd.DataFrame(data).dropna()
        df_clean['interaction'] = df_clean['initial_word_correct'] * \
            (1 - df_clean['forgetting_rate'])

        formula = 'correct_recall ~ initial_word_correct + forgetting_rate + interaction'
        model = smf.ols(formula, data=df_clean)
        result = model.fit()

        interaction_effect = result.params['interaction']
        ci = result.bse['interaction'] * 1.96

        avg_accuracy = df['correct_recall'].mean()
        init_error_rate = 1 - df['initial_word_correct'].mean()
        avg_forgetting = df['forgetting_rate'].mean()

        if ((storing_df['engine'] == engine) & (storing_df['run'] == run)).any():
            storing_df.loc[(storing_df['engine'] == engine) & (
                storing_df['run'] == run), 'performance_score1'] = avg_accuracy
            storing_df.loc[(storing_df['engine'] == engine) & (
                storing_df['run'] == run), 'behaviour_score1'] = init_error_rate
            storing_df.loc[(storing_df['engine'] == engine) & (
                storing_df['run'] == run), 'behaviour_score2'] = avg_forgetting
            storing_df.loc[(storing_df['engine'] == engine) & (
                storing_df['run'] == run), 'behaviour_score3'] = interaction_effect
            storing_df.loc[(storing_df['engine'] == engine) & (
                storing_df['run'] == run), 'behaviour_score3_CI'] = ci
        else:
            storing_df.loc[len(storing_df)] = [
                engine, run, avg_accuracy, 'serial memory accuracy',
                init_error_rate, 'initiation error rate',
                avg_forgetting, 'mean forgetting rate',
                interaction_effect, 'interaction effect', ci
            ]

        return storing_df


if __name__ == '__main__':
    StoringSerialMemoryScores().get_all_scores()
