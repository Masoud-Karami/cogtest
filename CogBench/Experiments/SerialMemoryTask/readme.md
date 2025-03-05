# Serial Recall Learning for LLMs

## Key Features of the Implementation

### Human_Based Task:

1- the participants were asked to learn 27 word lists until they could recite each list without error. The lists consisted of 7, 13, or 19 words selected randomly and without replacement from the noun subset of the `Toronto Word Pool`. Each list was learned using the procedure of alternating study and test trials, under either

    - **constant** the participants studied lists in the usual manner of multitrial serial recall. The presentation

of the list on each study trial was kept in the same order and began with the same word. The participants were asked to recall the list in the presented order.

or

    - **varied** (i.e., spin) the order of the list items was again kept constant. Each study trial, however, began with a randomly selected item in the list and continued in order from that item, wrapping around through all the items in the list. No item was used on more than one study–test trial to begin the list. The participants were asked to recall the list in the order presented on the most recent study trial.


2. Thus, the experiment had a $3 \times 2$ within-subjects design, with three list lengths (7, 13, and 19) and two starting position conditions (constant and spin).

3. The participants were tested in four separate sessions.
4. The practice session consisted of three lists for each starting position condition, one of each list length.
5. Each of the three subsequent test sessions consisted of nine lists from one of the two starting position conditions—three lists of each of the three lengths (7, 13, and 19) in random order. The order of the spin and control conditions alternated across sessions and was counterbalanced across participants.

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

- `performance_score1 (Accuracy)` Proportion of correctly recalled words.

- `behaviour_score1 (Initiation Errors)`: Frequency of incorrect first-word recalls.

- `behaviour_score2 (Intertrial Forgetting)`: Mean rate of forgetting between trials.