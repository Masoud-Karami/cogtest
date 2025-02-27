# Serial Recall Learning for LLMs

## Key Features of the Implementation

### Word Pool Loading

- The script loads words from a file (`toronto_word_pool.txt`), which should be placed in the same directory as the script.

### Randomized List Generation

- Each trial selects a unique word list to avoid repetition of the same list in multiple trials.

### Constant vs. Spin Conditions

- **Constant Condition**: The same list order is used for every trial.
- **Spin Condition**: A random word serves as the starting point, shifting the order of words in the list.

### LLM Recall and Evaluation

- The Large Language Model (LLM) is prompted with the study list and asked to recall it in the correct order.
- Accuracy is calculated as the proportion of correctly recalled words.
- If accuracy reaches 100%, the trial stops early.

### Experimental Control

- The experiment repeats over multiple runs, testing different list lengths and conditions to ensure robustness.

### Output DataFrame

- Results, including accuracy, trial number, and recalled words, are logged into a DataFrame for analysis.

## Conclusion

This design effectively replicates serial recall learning for LLMs while maintaining experimental integrity.

Let me know if you need modifications!

## Scoring are based on

`performance_score1 (Accuracy)` Proportion of correctly recalled words.
`behaviour_score1 (Initiation Errors)`: Frequency of incorrect first-word recalls.
`behaviour_score2 (Intertrial Forgetting)`: Mean rate of forgetting between trials.
