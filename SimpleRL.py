import numpy as np
import matplotlib.pyplot as plt

# Define the environment
N_STATES = 10
START = 5
GOAL = 9

# Define the policy
policy = np.zeros(N_STATES)
policy[GOAL] = 1

# Define the state values
values = np.zeros(N_STATES)
values[GOAL] = 1

# Define the reward function
reward = np.zeros(N_STATES)
reward[GOAL] = 1

# Define the transition probabilities
P = np.zeros((N_STATES, N_STATES))
for i in range(N_STATES):
    if i == GOAL:
        P[i, i] = 1
    elif i == 0:
        P[i, i] = 0.5
        P[i, i + 1] = 0.5
    elif i == N_STATES - 1:
        P[i, i] = 0.5
        P[i, i - 1] = 0.5
    else:
        P[i, i - 1] = 0.5
        P[i, i + 1] = 0.5

# Define the discount factor
GAMMA = 0.9

# Evaluate the policy
while True:
    new_values = np.zeros(N_STATES)
    for i in range(N_STATES):
        if i == GOAL:
            continue
        new_values[i] = reward[i] + GAMMA * np.dot(P[i, :], values)
    if np.abs(values - new_values).max() < 1e-9:
        break
    values = new_values

# Plot the state values
plt.plot(values)
plt.xlabel('State')
plt.ylabel('Value')
plt.show()
