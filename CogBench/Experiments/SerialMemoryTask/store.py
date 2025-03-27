import argparse
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # Allows importing CogBench as a package
from CogBench.base_classes import StoringScores


class StoringSerialMemoryScores(StoringScores):
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        """Add arguments for scoring."""
        self.parser.add_argument('--columns', nargs='+', default=[
            'performance_score1', 'performance_score1_name', 
            'behaviour_score1', 'behaviour_score1_name', 
            'behaviour_score2', 'behaviour_score2_name'
        ], help='List of columns to add to the CSV file')

    def get_scores(self, df, storing_df, engine, run):
        """
        Compute serial memory task scores.

        Args:
            df (pd.DataFrame): DataFrame with experiment results.
            storing_df (pd.DataFrame): DataFrame storing scores.
            engine (str): LLM engine name.
            run (int): Run identifier.

        Returns:
            pd.DataFrame: Updated DataFrame with computed scores.
        """
        # Ensure necessary columns exist
        for column in self.parser.parse_args().columns:
            if column not in storing_df.columns:
                storing_df[column] = np.nan

        # Compute scores
        total_trials = len(df)
        total_correct = df['correct_recall'].sum()  # Correctly recalled words
        initiation_errors = (df['initial_word_correct'] == 0).sum()  # Incorrect first word recall
        intertrial_forgetting = df['forgetting_rate'].mean()  # Avg forgetting rate between trials

        accuracy = total_correct / total_trials if total_trials > 0 else 0
        initiation_error_rate = initiation_errors / total_trials if total_trials > 0 else 0

        # Store scores in the CSV file
        if ((storing_df['engine'] == engine) & (storing_df['run'] == run)).any():
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'performance_score1'] = accuracy
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score1'] = initiation_error_rate
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score2'] = intertrial_forgetting
        else:
            storing_df.loc[len(storing_df)] = [
                engine, run, accuracy, 'serial memory accuracy', 
                initiation_error_rate, 'initiation errors', 
                intertrial_forgetting, 'intertrial forgetting'
            ]
        
        return storing_df


if __name__ == '__main__':
    StoringSerialMemoryScores().get_all_scores()
