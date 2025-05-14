import argparse
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from CogBench.base_classes import StoringScores

# Ensure proper import of CogBench as a package
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
        # if '--version_number' not in sys.argv:
        #     self.parser.add_argument(
        #         '--version_number', type=str, default='1', help='Version number of the dataset')

    def get_all_scores(self):
        args = self.parser.parse_args()

        scores_csv_name = f"scores_data{'V' + args.version_number if args.version_number != '1' else ''}.csv"
        data_folder = f"data{'V' + args.version_number if args.version_number != '1' else ''}"

        if 'all' in args.engines:
            engines = [file.split('.')[0] for file in os.listdir(data_folder)
                       if os.path.isfile(os.path.join(data_folder, file)) and file.endswith('.csv')]
        else:
            engines = args.engines

        # Initialize storing dataframe
        storing_df = pd.read_csv(scores_csv_name) if os.path.isfile(
            scores_csv_name) else pd.DataFrame(columns=args.columns + ['engine', 'run'])

        for engine in tqdm(engines):
            print(
                f'Fitting for engine: {engine}-------------------------------------------')
            path = os.path.join(data_folder, f"{engine}.csv")
            full_df = pd.read_csv(path)

            # Support for participant-wise logic, currently all as participant 0
            full_df['participant'] = 0
            no_participants = full_df['participant'].max() + 1

            for participant in range(no_participants):
                df_run = full_df[full_df['participant']
                                 == participant].reset_index(drop=True)
                storing_df = self.get_scores(
                    df_run, storing_df, engine, run=participant)
                storing_df.to_csv(scores_csv_name, index=False)

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

        # --- Behavioral Metrics ---
        initiation_error_rate = 1 - df['init_correct'].mean()
        forgetting_rate = df['forget_rate'].mean()

        # Serial position effects
        primacy_correct = 0
        recency_correct = 0
        primacy_total = 0
        recency_total = 0

        for _, row in df.iterrows():
            study = row['study_list'].split(',')
            recalled_raw = row['recalled_list']
            # [['a', 'b', 'c', 'd', 'e', 'f', 'g'], [], ['a', 'b', 'c', 'd', 'e', 'x', 'g']]
            if not isinstance(recalled_raw, str) or pd.isna(recalled_raw):
                recall = []
            else:
                recall = recalled_raw.split(',')

            # if len(study) < 4 or len(recall) != len(study):
            #     continue

            # Remove strict length filter — keep all trials, even imperfect ones
            # Align lengths for scoring (e.g., primacy/recency)
            while len(recall) < len(study):
                recall.append("")  # pad missing with empty
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
            # storing_df.loc[len(storing_df)] = [
            #     engine, 'engine',
            #     run, 'run',
            #     accuracy, 'serial memory accuracy',
            #     ttc_mean, 'mean TTC',
            #     initiation_error_rate, 'initiation error rate',
            #     forgetting_rate, 'intertrial forgetting',
            #     primacy_effect, 'primacy effect',
            #     recency_effect, 'recency effect'
            # ]
            storing_df.loc[len(storing_df)] = [
                engine, run,
                accuracy, 'serial memory accuracy',
                ttc_mean, 'mean TTC',
                initiation_error_rate, 'initiation error rate',
                forgetting_rate, 'intertrial forgetting',
                primacy_effect, 'primacy effect',
                recency_effect, 'recency effect'
            ]
            # Debugging information: remove the above block and uncomment the block before taht to see the DataFrame structure
            # print("\n--- DEBUG INFO ---")
            # print("DataFrame columns:", storing_df.columns.tolist())
            # print("Length of columns:", len(storing_df.columns))
            # row_data = [
            #     engine, 'engine',
            #     run, 'run',
            #     accuracy, 'serial memory accuracy',
            #     ttc_mean, 'mean TTC',
            #     initiation_error_rate, 'initiation error rate',
            #     forgetting_rate, 'intertrial forgetting',
            #     primacy_effect, 'primacy effect',
            #     recency_effect, 'recency effect'
            # ]
            # print("Row to insert:", row_data)
            # print("Length of row_data:", len(row_data))
            # storing_df.loc[len(storing_df)] = row_data

        # --- Print Behavioral Metrics to Shell --- for test experiments without saving with gpt-3
        print(f"\n--- Behavioral Metrics for engine: {engine}, run: {run} ---")
        print(f"  Initiation Error Rate:     {initiation_error_rate:.3f}")
        print(f"  Intertrial Forgetting:     {forgetting_rate:.3f}")
        print(f"  Primacy Effect:            {primacy_effect:.3f}")
        print(f"  Recency Effect:            {recency_effect:.3f}")
        print("------------------------------------------------------\n")

        return storing_df


if __name__ == '__main__':
    StoringSerialMemoryScores().get_all_scores()


# TODO add these!
   # def relative_order_scoring(self, recalled, study_list):
    #     return sum(
    #         study_list.index(
    #             recalled[i]) + 1 == study_list.index(recalled[i + 1])
    #         for i in range(len(recalled) - 1)
    #         if recalled[i] in study_list and recalled[i + 1] in study_list
    #     )

    # def compute_forgetting_rate(self, prev_recall, current_recall):
    #     if not prev_recall:
    #         return np.nan
    #     return 1 - sum(1 for a, b in zip(prev_recall, current_recall) if a == b) / len(prev_recall)
