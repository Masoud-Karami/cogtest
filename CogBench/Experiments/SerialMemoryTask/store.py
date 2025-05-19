# -------------------------------------------------------------------------
# Next Section: store.py - Serial Memory Task Scoring & Behavioral Metrics
# -------------------------------------------------------------------------

from CogBench.base_classes import StoringScores
import argparse
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))


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

    def get_all_scores(self):
        args = self.parser.parse_args()

        scores_csv_name = f"scores_data{'V' + args.version_number if args.version_number != '1' else ''}.csv"
        data_folder = f"data{'V' + args.version_number if args.version_number != '1' else ''}"

        if 'all' in args.engines:
            engines = [file.split('.')[0] for file in os.listdir(data_folder)
                       if os.path.isfile(os.path.join(data_folder, file)) and file.endswith('.csv')]
        else:
            engines = args.engines

        storing_df = pd.read_csv(scores_csv_name) if os.path.isfile(
            scores_csv_name) else pd.DataFrame(columns=args.columns + ['engine', 'run'])

        for engine in tqdm(engines):
            print(f'Fitting for engine: {engine} --------------------------')
            path = os.path.join(data_folder, f"{engine}.csv")
            full_df = pd.read_csv(path)
            full_df['participant'] = 0
            no_participants = full_df['participant'].max() + 1

            for participant in range(no_participants):
                df_run = full_df[full_df['participant'] ==
                                 participant].reset_index(drop=True)
                storing_df = self.get_scores(
                    df_run, storing_df, engine, run=participant)
                storing_df.to_csv(scores_csv_name, index=False)

    def get_scores(self, df, storing_df, engine, run):
        args = self.parser.parse_args()
        for col in args.columns:
            if col not in storing_df.columns:
                storing_df[col] = np.nan

        total_words = df['list_length'].sum()
        accuracy = df['correct_recall'].sum(
        ) / total_words if total_words > 0 else 0

        ttc_df = df[df['ttc_achieved'] == True].groupby(
            ['session', 'condition', 'list_index'])['trial'].min()
        ttc_mean = ttc_df.mean() if len(ttc_df) > 0 else np.nan

        initiation_error_rate = 1 - df['init_correct'].mean()
        forgetting_rate = df['forget_rate'].mean()

        primacy_correct = 0
        recency_correct = 0
        primacy_total = 0
        recency_total = 0

        for _, row in df.iterrows():
            study = row['study_list'].split(',')
            recalled_raw = row['recalled_list']
            if not isinstance(recalled_raw, str) or pd.isna(recalled_raw):
                recall = []
            else:
                recall = recalled_raw.split(',')

            while len(recall) < len(study):
                recall.append("")
            if len(recall) > len(study):
                recall = recall[:len(study)]

            primacy_correct += int(recall[0] == study[0]) + \
                int(recall[1] == study[1])
            primacy_total += 2
            recency_correct += int(recall[-2] == study[-2]) + \
                int(recall[-1] == study[-1])
            recency_total += 2

        primacy_effect = primacy_correct / primacy_total if primacy_total else np.nan
        recency_effect = recency_correct / recency_total if recency_total else np.nan

        existing = (storing_df['engine'] == engine) & (
            storing_df['run'] == run)

        if existing.any():
            storing_df.loc[existing, 'performance_score1'] = accuracy
            storing_df.loc[existing,
                           'performance_score1_name'] = 'serial memory accuracy'
            storing_df.loc[existing, 'performance_score2'] = ttc_mean
            storing_df.loc[existing, 'performance_score2_name'] = 'mean TTC'
            storing_df.loc[existing,
                           'behaviour_score1'] = initiation_error_rate
            storing_df.loc[existing,
                           'behaviour_score1_name'] = 'initiation error rate'
            storing_df.loc[existing, 'behaviour_score2'] = forgetting_rate
            storing_df.loc[existing,
                           'behaviour_score2_name'] = 'intertrial forgetting'
            storing_df.loc[existing, 'behaviour_score3'] = primacy_effect
            storing_df.loc[existing,
                           'behaviour_score3_name'] = 'primacy effect'
            storing_df.loc[existing, 'behaviour_score4'] = recency_effect
            storing_df.loc[existing,
                           'behaviour_score4_name'] = 'recency effect'
        else:
            storing_df.loc[len(storing_df)] = [
                engine, run,
                accuracy, 'serial memory accuracy',
                ttc_mean, 'mean TTC',
                initiation_error_rate, 'initiation error rate',
                forgetting_rate, 'intertrial forgetting',
                primacy_effect, 'primacy effect',
                recency_effect, 'recency effect'
            ]

        print(f"\n--- Behavioral Metrics for engine: {engine}, run: {run} ---")
        print(f"  Initiation Error Rate:     {initiation_error_rate:.3f}")
        print(f"  Intertrial Forgetting:     {forgetting_rate:.3f}")
        print(f"  Primacy Effect:            {primacy_effect:.3f}")
        print(f"  Recency Effect:            {recency_effect:.3f}")
        print("------------------------------------------------------\n")

        return storing_df


if __name__ == '__main__':
    StoringSerialMemoryScores().get_all_scores()
