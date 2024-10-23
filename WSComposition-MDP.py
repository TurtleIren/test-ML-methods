# Value Iteration Algorithm
import numpy as np

# Визначаємо параметри MDP
states = ['s1', 's2', 's3']  # можливі стани
actions = ['a1', 'a2']  # можливі дії
transition_probs = {  # ймовірності переходів між станами
    's1': {'a1': [('s2', 1.0)], 'a2': [('s3', 1.0)]},
    's2': {'a1': [('s1', 0.5), ('s3', 0.5)], 'a2': [('s3', 1.0)]},
    's3': {'a1': [('s1', 1.0)], 'a2': [('s2', 1.0)]}
}
rewards = {  # винагороди для кожної дії у кожному стані
    's1': {'a1': 10, 'a2': 5},
    's2': {'a1': -1, 'a2': 2},
    's3': {'a1': 0, 'a2': -10}
}
gamma = 0.9  # коефіцієнт дисконтування
threshold = 0.01  # поріг для зупинки ітерацій

# Ініціалізація функції значення
values = {s: 0 for s in states}

# Value Iteration Algorithm
def value_iteration(states, actions, transition_probs, rewards, gamma, threshold):
    while True:
        delta = 0
        for state in states:
            v = values[state]
            new_values = []
            for action in actions:
                total = sum([p * (rewards[state][action] + gamma * values[next_state])
                             for next_state, p in transition_probs[state][action]])
                new_values.append(total)
            values[state] = max(new_values)
            delta = max(delta, abs(v - values[state]))
        if delta < threshold:
            break
    return values

optimal_values = value_iteration(states, actions, transition_probs, rewards, gamma, threshold)
print("Optimal Values:", optimal_values)

#Результат роботи програми:
#Optimal Values: {'s1': 43.50304889327011, 's2': 37.233780258291354, 's3': 39.1527440039431}
