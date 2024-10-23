import numpy as np
import time

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

# Policy Evaluation (оцінка політики для Policy Iteration)
def policy_evaluation(policy, states, actions, transition_probs, rewards, gamma, threshold):
    values = {s: 0 for s in states}
    while True:
        delta = 0
        for state in states:
            v = values[state]
            action = policy[state]
            new_value = sum([p * (rewards[state][action] + gamma * values[next_state])
                             for next_state, p in transition_probs[state][action]])
            values[state] = new_value
            delta = max(delta, abs(v - values[state]))
        if delta < threshold:
            break
    return values

# Policy Iteration Algorithm
def policy_iteration(states, actions, transition_probs, rewards, gamma, threshold):
    policy = {s: actions[0] for s in states}  # Ініціалізуємо політику випадково
    while True:
        values = policy_evaluation(policy, states, actions, transition_probs, rewards, gamma, threshold)
        policy_stable = True
        for state in states:
            old_action = policy[state]
            new_values = []
            for action in actions:
                total = sum([p * (rewards[state][action] + gamma * values[next_state])
                             for next_state, p in transition_probs[state][action]])
                new_values.append(total)
            best_action = actions[np.argmax(new_values)]
            if old_action != best_action:
                policy_stable = False
            policy[state] = best_action
        if policy_stable:
            break
    return policy, values

# Iterative Policy Evaluation (статична оцінка політики)
def iterative_policy_evaluation(states, actions, transition_probs, rewards, gamma, threshold):
    policy = {s: actions[0] for s in states}
    values = policy_evaluation(policy, states, actions, transition_probs, rewards, gamma, threshold)
    return values

# Відстеження часу виконання
start_time = time.time()
optimal_values = value_iteration(states, actions, transition_probs, rewards, gamma, threshold)
value_iteration_time = time.time() - start_time

start_time = time.time()
optimal_policy, policy_values = policy_iteration(states, actions, transition_probs, rewards, gamma, threshold)
policy_iteration_time = time.time() - start_time

start_time = time.time()
iterative_policy_values = iterative_policy_evaluation(states, actions, transition_probs, rewards, gamma, threshold)
iterative_policy_eval_time = time.time() - start_time

# Виведення результатів
print("Optimal Values (Value Iteration):", optimal_values)
print("Time for Value Iteration: {:.5f} seconds".format(value_iteration_time))

print("\nOptimal Policy (Policy Iteration):", optimal_policy)
print("Optimal Values (Policy Iteration):", policy_values)
print("Time for Policy Iteration: {:.5f} seconds".format(policy_iteration_time))

print("\nValues (Iterative Policy Evaluation):", iterative_policy_values)
print("Time for Iterative Policy Evaluation: {:.5f} seconds".format(iterative_policy_eval_time))
