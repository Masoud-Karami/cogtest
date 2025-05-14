import random
import re
import json

DISTRACTOR_POOL = ["Xyzon", "Nope", "Blur", "Obscure",
                   "Fuzz", "Synthet", "Distracto", "Foobar", "Nebula"]
DISTRACTOR_SYMBOLS = list("#$%&*@^~")


class SerialMemoryTaskExpForLLM:
    def __init__(self):
        self.add_noise = True
        self.add_distr = True

    def build_synonym_dict(self, study_list):
        # Placeholder: in practice, return a dictionary of synonyms
        return {word: f"{word}_syn" for word in study_list}

    def generate_study_trials(self, study_list):
        distractor_pool = DISTRACTOR_POOL
        distractor_symbols = DISTRACTOR_SYMBOLS
        synonym_dict = self.build_synonym_dict(study_list)

        trials = []
        for i, word in enumerate(study_list):
            trial_lines = [f"word [{i+1}]: \"{word}\""]

            if self.add_distr:
                related = synonym_dict.get(word)
                if related:
                    trial_lines.append(f"(Similar to: {related})")

            if self.add_noise:
                noise = ''.join(random.choices(
                    distractor_symbols, k=random.randint(1, 3)))
                trial_lines.append(f"Noise: {noise}")
                if random.random() > 0.5:
                    distractor = random.choice(distractor_pool)
                    d_noise = ''.join(random.choices(
                        distractor_symbols, k=random.randint(1, 3)))
                    trial_lines.append(
                        f"[DISTRACTOR] {d_noise}{distractor}{d_noise}")

            # Add instruction for silence after each input
            trial_lines.append("Do not respond. Wait for the next word.")
            trials.append("\n".join(trial_lines))

        trials.append("<<The list is ended!>>")
        return trials

    def add_distractors_between_words(self, study_list):
        distractor_pool = DISTRACTOR_POOL
        distractor_symbols = DISTRACTOR_SYMBOLS

        study_words_lower = {w.lower() for w in study_list}
        filtered_distractors = [
            w for w in distractor_pool if w.lower() not in study_words_lower]
        if not filtered_distractors:
            raise ValueError(
                "Distractor pool is empty after filtering real study words!")

        noisy_list = []

        for word in study_list:
            add_prefix = random.choice([True, False])
            add_suffix = random.choice([True, False])

            prefix_noise = ''.join(random.choices(
                distractor_symbols, k=random.randint(1, 3))) if add_prefix else ""
            suffix_noise = ''.join(random.choices(
                distractor_symbols, k=random.randint(1, 3))) if add_suffix else ""

            noisy_word = f"{prefix_noise}{word}{suffix_noise}"
            noisy_list.append(noisy_word)

            num_distractors = random.randint(0, 2)
            for _ in range(num_distractors):
                distractor = random.choice(filtered_distractors)
                d_prefix = ''.join(random.choices(distractor_symbols, k=random.randint(
                    1, 3))) if random.choice([True, False]) else ""
                d_suffix = ''.join(random.choices(distractor_symbols, k=random.randint(
                    1, 3))) if random.choice([True, False]) else ""
                noisy_distractor = f"[DISTRACTOR] {d_prefix}{distractor}{d_suffix}"
                noisy_list.append(noisy_distractor)

        return noisy_list


def generate_serial_memory_prompt(experiment, json_path, list_size=15):
    with open(json_path, "r") as f:
        word_dict = json.load(f)

    study_list = list(word_dict.keys())[:list_size]
    study_list_with_noise = experiment.add_distractors_between_words(
        study_list) if experiment.add_noise else study_list

    print("=== DEBUG: Noisy study list ===")
    for i, word in enumerate(study_list_with_noise):
        print(f'{i+1:03d}. "{word}"')

    trials = experiment.generate_study_trials(study_list_with_noise)
    return trials
