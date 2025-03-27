import argparse
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from CogBench.base_classes import StoringScores


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))  # allows to import CogBench as a package

print("PYTHONPATH set to:", os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")))


class StoringSerialMemoryScores(StoringScores):
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        self.parser.add_argument('--columns', nargs='+', default=[
            'performance_score1', 'performance_score1_name',
            'performance_score2', 'performance_score2_name',
            'behaviour_score1', 'behaviour_score1_name',
            'behaviour_score2', 'behaviour_score2_name',
            'behaviour_score3', 'behaviour_score3_name',
            'behaviour_score4', 'behaviour_score4_name'
        ])

    def get_scores(self, df, storing_df, engine, run):
        args = self.parser.parse_args()
        for col in args.columns:
            if col not in storing_df.columns:
                storing_df[col] = np.nan

        # --- Performance Metrics ---
        total_words = df['list_length'].sum()
        accuracy = df['correct_recall'].sum(
        ) / total_words if total_words > 0 else 0

        ttc_df = df[df['ttc_achieved'] == True].groupby(
            ['session', 'condition', 'list_index'])['trial'].min()
        ttc_mean = ttc_df.mean() if len(ttc_df) > 0 else np.nan
        ttc_achieved_ratio = df[df['ttc_achieved'] == True][['session', 'condition', 'list_index']].drop_duplicates().shape[0] \
            / df[['session', 'condition', 'list_index']].drop_duplicates().shape[0]

        # --- Behavioral Metrics ---
        initiation_error_rate = 1 - df['initial_word_correct'].mean()
        forgetting_rate = df['forgetting_rate'].mean()

        # Serial position effects
        primacy_correct = 0
        recency_correct = 0
        primacy_total = 0
        recency_total = 0

        for _, row in df.iterrows():
            study = row['study_list'].split(',')
            recall = row['recalled_list'].split(',')

            if len(study) < 4 or len(recall) != len(study):
                continue

            primacy_correct += int(recall[0] == study[0]) + \
                int(recall[1] == study[1])
            primacy_total += 2
            recency_correct += int(recall[-2] == study[-2]) + \
                int(recall[-1] == study[-1])
            recency_total += 2

        primacy_effect = primacy_correct / primacy_total if primacy_total else np.nan
        recency_effect = recency_correct / recency_total if recency_total else np.nan

        # --- Store results ---
        existing = (storing_df['engine'] == engine) & (
            storing_df['run'] == run)
        row_data = [
            engine, run,
            accuracy, 'serial memory accuracy',
            ttc_mean, 'mean TTC',
            initiation_error_rate, 'initiation error rate',
            forgetting_rate, 'intertrial forgetting',
            primacy_effect, 'primacy effect',
            recency_effect, 'recency effect'
        ]

        if existing.any():
            storing_df.loc[existing, args.columns] = row_data[2:]
        else:
            storing_df.loc[len(storing_df)] = row_data

        return storing_df


if __name__ == '__main__':
    StoringSerialMemoryScores().get_all_scores()
