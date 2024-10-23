import numpy as np
import random

# Ініціалізуємо параметри середовища
states = ['s1', 's2', 's3']  # можливі стани
actions = ['a1', 'a2']  # можливі дії
q_table = np.zeros((len(states), len(actions)))  # Q-таблиця
alpha = 0.1  # швидкість навчання
gamma = 0.9  # коефіцієнт дисконтування
epsilon = 0.1  # параметр ε для ε-greedy політики
episodes = 1000  # кількість епізодів

# Визначаємо винагороди для дій у кожному стані
rewards = {  # винагороди для кожної дії
    's1': {'a1': 10, 'a2': 5},
    's2': {'a1': -1, 'a2': 2},
    's3': {'a1': 0, 'a2': -10}
}

# Функція для вибору дії відповідно до ε-greedy політики
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(q_table[states.index(state)])]

# Функція для оновлення Q-значень
def update_q_table(state, action, reward, next_state):
    state_idx = states.index(state)
    action_idx = actions.index(action)
    next_state_idx = states.index(next_state)
    best_next_action = np.argmax(q_table[next_state_idx])
    q_table[state_idx, action_idx] += alpha * (reward + gamma * q_table[next_state_idx, best_next_action] - q_table[state_idx, action_idx])

# Навчання з використанням Q-learning
for episode in range(episodes):
    state = random.choice(states)
    done = False
    
    while not done:
        action = choose_action(state, epsilon)
        next_state = random.choice(states)  # випадковий перехід до нового стану
        reward = rewards[state][action]
        update_q_table(state, action, reward, next_state)
        
        state = next_state
        if state == 's3':  # термінальний стан
            done = True

# Виведення Q-таблиці
print("Q-table after training:")
print(q_table)

#Як і в попередньому прикладі, використовуємо набір вебсервісів з атрибутами QoS:
#[
#    {"name": "WebService1", "availability": 0.99, "execution_time": 1.2, "throughput": 50},
#    {"name": "WebService2", "availability": 0.95, "execution_time": 2.0, "throughput": 40},
#    {"name": "WebService3", "availability": 0.98, "execution_time": 1.5, "throughput": 45}
#]
#Отриманий результат:
#Q-table after training:
#[[47.37387928 40.80646288]
# [35.46946697 40.14520716]
# [38.7222855  17.36345372 ]]

