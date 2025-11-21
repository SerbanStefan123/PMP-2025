import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

states = ["W", "R", "R"]
n_states = len(states)

observations = ["L", "M", "H"]
n_observations = len(observations)

start_probability = np.array([0.4, 0.3, 0.3])

transition_probability = np.array([
    [0.6, 0.3, 0.1], 
    [0.2, 0.7, 0.1],  
    [0.3, 0.2, 0.5],      
])

emission_probability = np.array([
    [0.1, 0.7, 0.2],  
    [0.05, 0.25, 0.7],  
    [0.8, 0.15, 0.05]
])


model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

#Ca in seminar
observations_sequence = np.array([1,2,0]).reshape(-1, 1)
hidden_states = model.predict(observations_sequence)
logprob = model.score(observations_sequence)
print(np.exp(logprob))
print("Most likely hidden states:", hidden_states)

#Aici folosim algoritmul viterbi
logprob_vit, hidden_states = model.decode(observations_sequence, algorithm="viterbi")
print("Most likely hidden states (indices):", hidden_states)
print("Most likely hidden states (names):", [states[i] for i in hidden_states])
print("Log P(path*, observations):", logprob_vit)
print("P(path*, observations):", np.exp(logprob_vit))

sns.set_style("darkgrid")
plt.plot(hidden_states, '-o', label="Hidden State")
plt.xlabel("Time Step")
plt.ylabel("Hidden State (Ce Dificultate)")
plt.yticks(ticks=range(n_states), labels=states)
plt.legend()
plt.title("Predicted Hidden States Over Time")
plt.show()
