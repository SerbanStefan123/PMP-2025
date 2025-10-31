import numpy as np
from hmmlearn import hmm

states = ["Difficult", "Medium", "Easy"]
n_states = len(states)

grades = ["FB", "B", "S", "NS"]
n_observations = len(grades)

start_probability = np.array([1 / 3, 1 / 3, 1 / 3])

transition_probability = np.array(
    [
        [0.0, 0.5, 0.5],
        [0.5, 0.25, 0.25],
        [0.5, 0.25, 0.25],
    ]
)

emission_probability = np.array(
    [
        [0.1, 0.2, 0.4, 0.3],
        [0.15, 0.25, 0.5, 0.1],
        [0.2, 0.3, 0.4, 0.1],
    ]
)

model = hmm.CategoricalHMM(n_components=n_states, init_params="")
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability


obs_map = {"FB": 0, "B": 1, "S": 2, "NS": 3}
observations = np.array(
    [obs_map[x] for x in ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B"]]
).reshape(-1, 1)

logprob = model.score(observations)
print(f"Log Probability of sequence: {logprob:.3f}")


hidden_states = model.predict(observations)
most_probable_sequence = [states[i] for i in hidden_states]
print("Most probable sequence of test difficulties:")
print(most_probable_sequence)

viterbi_logprob = model.score(observations)
print(f"Log-probability of most probable sequence: {viterbi_logprob:.6f}")
