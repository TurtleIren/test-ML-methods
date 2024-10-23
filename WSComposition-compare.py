import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt

class QLearningWSC:
    def __init__(self, service_classes, flows, service_data, gamma=0.8, alpha=0.1, epsilon=0.2):
        self.service_classes = service_classes  # Класи веб-сервісів
        self.flows = flows  # Набори можливих флоу
        self.service_data = service_data  # Дані веб-сервісів (QoS параметри)
        self.gamma = gamma  # Дисконтуючий фактор
        self.alpha = alpha  # Швидкість навчання
        self.epsilon = epsilon  # Ступінь дослідження
        self.q_table = {}  # Таблиця Q-значень

        # Ініціалізація станів і дій для кожного флоу
        self.states = self.generate_states()
        self.actions = self.generate_actions()

    def generate_states(self):
        """Генерує всі можливі стани для кожного флоу."""
        states = []
        for i, flow in enumerate(self.flows):
            for step in range(len(flow) + 1):  # Враховуємо всі стани флоу, включаючи кінцевий
                states.append(f"flow_{i}_state_{step}")
        return states

    def generate_actions(self):
        """Генерує дії - вибір сервісу з кожного класу для кожного флоу."""
        actions = {}
        for i, flow in enumerate(self.flows):
            for step, service_class in enumerate(flow):
                state = f"flow_{i}_state_{step}"
                # Дії для поточного стану: всі сервіси з відповідного класу
                actions[state] = self.service_classes[service_class]
            
            # Останній стан не має жодних дій
            final_state = f"flow_{i}_state_{len(flow)}"
            actions[final_state] = []  # Кінцевий стан без дій
        return actions

    def generate_composite_service(self, flow_idx):
        """Генерує композитний веб-сервіс для конкретного флоу на основі навченої політики."""
        state = f"flow_{flow_idx}_state_0"
        flow = []
        for step in range(len(self.flows[flow_idx])):
            action = self.choose_action(state)
            flow.append(action)
            state = f"flow_{flow_idx}_state_{step + 1}"
        return flow

    def get_reward(self, action):
        """Отримує винагороду для конкретної дії на основі явно заданих значень QoS."""
        # Витягуємо індекси класу та сервісу на основі назви сервісу
        for class_idx, services in self.service_classes.items():
            if action in services:
                service_idx = services.index(action)
                break
        
        service = self.service_data[class_idx - 1][service_idx]  # -1, щоб правильно отримати клас

        # Витягаємо атрибути QoS для цього веб-сервісу
        availability = service['availability']
        exec_time = service['exec_time']
        throughput = service['throughput']
        
        # Формула винагороди на основі QoS
        reward = (availability * 0.4) - (exec_time * 0.4) + (throughput * 0.2)
        return reward

    def choose_action(self, state):
        """Вибирає дію за epsilon-greedy стратегією."""
        if state not in self.actions:
            raise KeyError(f"State {state} не ініціалізований у словнику дій")

        # Якщо немає доступних дій у кінцевому стані, повертаємо None
        if not self.actions[state]:
            return None

        if random.uniform(0, 1) < self.epsilon:
            # Дослідження: вибираємо випадковий веб-сервіс
            return random.choice(self.actions[state])
        else:
            # Експлуатація: вибираємо дію з максимальним Q-значенням
            if state not in self.q_table:
                return random.choice(self.actions[state])
            return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, state, action, reward, next_state):
        """Оновлює Q-значення для поточного стану на основі дії, нагороди та наступного стану."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0}

        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table.get(next_state, {}).values(), default=0)

        # Q-learning формула оновлення
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def train(self, episodes=1000):
        """Навчання алгоритму протягом визначеної кількості епізодів."""
        for episode in range(episodes):
            for flow_idx, flow in enumerate(self.flows):
                state = f"flow_{flow_idx}_state_0"  # Початковий стан для кожного флоу
                for step in range(len(flow)):
                    action = self.choose_action(state)
                    if action is None:  # Якщо ми у кінцевому стані
                        break

                    reward = self.get_reward(action)
                    next_state = f"flow_{flow_idx}_state_{step + 1}"

                    # Оновлення Q-таблиці
                    self.update_q_table(state, action, reward, next_state)

                    state = next_state
                    if state == f"flow_{flow_idx}_state_{len(flow)}":
                        break

class SARSALearningWSC(QLearningWSC):
    def update_q_table(self, state, action, reward, next_state, next_action):
        # Ініціалізація стану в Q-таблиці, якщо він не існує
        if state not in self.q_table:
            self.q_table[state] = {action: 0}

        # Ініціалізація наступного стану, якщо його немає
        if next_state not in self.q_table:
            self.q_table[next_state] = {}

        # Ініціалізація дії, якщо її немає
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        
        if next_action not in self.q_table[next_state]:
            self.q_table[next_state][next_action] = 0
        
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state].get(next_action, 0)
        
        # Оновлення Q-значення
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def train(self, episodes=1000):
        for episode in range(episodes):
            for flow_idx, flow in enumerate(self.flows):
                state = f"flow_{flow_idx}_state_0"
                action = self.choose_action(state)
                for step in range(len(flow)):
                    reward = self.get_reward(action)
                    next_state = f"flow_{flow_idx}_state_{step + 1}"
                    next_action = self.choose_action(next_state)

                    self.update_q_table(state, action, reward, next_state, next_action)

                    state = next_state
                    action = next_action
                    if state == f"flow_{flow_idx}_state_{len(flow)}":
                        break

class ValueIterationWSC(QLearningWSC):
    def value_iteration(self, iterations=1000):
        """Алгоритм Value Iteration"""
        for i in range(iterations):
            new_value_table = {}
            for state in self.states:
                if not self.actions.get(state):
                    # Якщо стан - кінцевий, не має доступних дій
                    new_value_table[state] = 0
                    continue

                value_list = []
                for action in self.actions[state]:
                    reward = self.get_reward(action)
                    next_state = self.get_next_state(state, action)

                    if isinstance(self.q_table.get(next_state, 0), dict):
                        max_next_q = max(self.q_table.get(next_state, {}).values(), default=0)
                    else:
                        max_next_q = self.q_table.get(next_state, 0)

                    value_list.append(reward + self.gamma * max_next_q)

                if value_list:
                    new_value_table[state] = max(value_list)
                else:
                    new_value_table[state] = 0

            self.q_table = new_value_table

    def choose_action(self, state):
        if state not in self.actions or not self.actions[state]:
            return None

        if state not in self.q_table:
            return random.choice(self.actions[state])

        if isinstance(self.q_table[state], dict):
            return max(self.q_table[state], key=self.q_table[state].get)
        else:
            return None

    def get_next_state(self, state, action):
        """Отримує наступний стан на основі поточного стану та дії."""
        flow_idx, step_idx = map(int, state.replace("flow_", "").replace("_state_", " ").split())
        next_step = step_idx + 1
        next_state = f"flow_{flow_idx}_state_{next_step}"
        return next_state if next_state in self.states else None

    def choose_action(self, state):
        """Вибирає дію на основі таблиці значень для Value Iteration."""
        if state not in self.actions:
            raise KeyError(f"State {state} не ініціалізований у словнику дій")

        # Якщо стан має порожній список дій, це кінцевий стан
        if not self.actions[state]:
            return None

        # Перевіряємо, чи є стан в таблиці q_table, і якщо там збережене не число, а словник
        if isinstance(self.q_table.get(state, 0), dict):
            return max(self.q_table[state], key=self.q_table[state].get)
        else:
            # Якщо це кінцевий стан (float в q_table), то немає більше дій
            return None

    def generate_composite_service(self, flow_idx):
        """Генерує композитний веб-сервіс для конкретного флоу на основі політики Value Iteration."""
        state = f"flow_{flow_idx}_state_0"
        flow = []
        for step in range(len(self.flows[flow_idx])):
            action = self.choose_action(state)
            if action is None:  # Якщо це кінцевий стан, зупиняємося
                break
            flow.append(action)
            state = f"flow_{flow_idx}_state_{step + 1}"
        return flow

class GreedyWSC:
    def __init__(self, service_classes, flows, service_data):
        self.service_classes = service_classes  # Класи веб-сервісів
        self.flows = flows  # Набори можливих флоу
        self.service_data = service_data  # Дані веб-сервісів (QoS параметри)

    def get_reward(self, service_class, service_idx):
        service = self.service_data[service_class - 1][service_idx]
        availability = service['availability']
        exec_time = service['exec_time']
        throughput = service['throughput']

        reward = (availability * 0.4) - (exec_time * 0.4) + (throughput * 0.2)
        return reward

    def find_flows_and_rewards(self):
        all_flows = []
        all_rewards = []
        
        for flow_idx, flow in enumerate(self.flows):
            total_reward = 0
            current_flow = []
            
            for service_class in flow:
                best_service_idx = max(range(len(self.service_classes[service_class])),
                                       key=lambda idx: self.get_reward(service_class, idx))
                current_flow.append(self.service_classes[service_class][best_service_idx])
                total_reward += self.get_reward(service_class, best_service_idx)
            
            all_flows.append(current_flow)
            all_rewards.append(total_reward)

        return all_flows, all_rewards

    def find_optimal_flow(self):
        all_flows, all_rewards = self.find_flows_and_rewards()
        optimal_idx = all_rewards.index(max(all_rewards))
        return all_flows[optimal_idx], all_rewards[optimal_idx]

class GeneticAlgorithmWSC:
    def __init__(self, service_classes, flows, service_data, population_size=20, generations=100, crossover_prob=0.8, mutation_prob=0.1):
        self.service_classes = service_classes  # Класи веб-сервісів
        self.flows = flows  # Набори можливих флоу
        self.service_data = service_data  # Дані веб-сервісів (QoS параметри)
        self.population_size = population_size  # Розмір популяції
        self.generations = generations  # Кількість поколінь
        self.crossover_prob = crossover_prob  # Ймовірність схрещування
        self.mutation_prob = mutation_prob  # Ймовірність мутації
        self.population = []  # Поточна популяція

    def initialize_population(self, flow_idx):
        """Ініціалізує популяцію випадковими рішеннями."""
        population = []
        flow = self.flows[flow_idx]
        for _ in range(self.population_size):
            chromosome = []
            for service_class in flow:
                service = random.choice(self.service_classes[service_class])
                chromosome.append(service)
            population.append(chromosome)
        self.population = population

    def get_reward(self, service_class, service_name):
        """Отримує винагороду для конкретного веб-сервісу на основі явно заданих значень QoS."""
        # Перевіряємо, чи належить сервіс до відповідного класу
        if service_name not in self.service_classes[service_class]:
            raise ValueError(f"Сервіс '{service_name}' не належить до класу {service_class}")

        service_idx = self.service_classes[service_class].index(service_name)
        service = self.service_data[service_class - 1][service_idx]
        availability = service['availability']
        exec_time = service['exec_time']
        throughput = service['throughput']

        reward = (availability * 0.4) - (exec_time * 0.4) + (throughput * 0.2)
        return reward

    def fitness(self, chromosome, flow_idx):
        """Обчислює фітнес (винагороду) для хромосоми (рішення)."""
        flow = self.flows[flow_idx]
        total_reward = 0
        for i, service in enumerate(chromosome):
            # Використовуємо поточний клас у флоу для правильного підбору винагороди
            total_reward += self.get_reward(flow[i], service)
        return total_reward

    def select_parents(self, flow_idx):
        """Відбирає двох батьків для схрещування (турнірний відбір)."""
        tournament_size = 3
        selected = random.sample(self.population, tournament_size)
        parent1 = max(selected, key=lambda chromo: self.fitness(chromo, flow_idx))  # Використовуємо flow_idx
        selected = random.sample(self.population, tournament_size)
        parent2 = max(selected, key=lambda chromo: self.fitness(chromo, flow_idx))  # Використовуємо flow_idx
        return parent1, parent2

    def crossover(self, parent1, parent2, flow_idx):
        """Схрещування двох батьків."""
        flow = self.flows[flow_idx]  # Визначаємо поточний флоу
        if random.uniform(0, 1) < self.crossover_prob:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
        else:
            child1, child2 = parent1, parent2

        # Переконуємось, що після схрещування всі сервіси належать до правильного класу
        for i in range(len(child1)):
            if child1[i] not in self.service_classes[flow[i]]:
                child1[i] = random.choice(self.service_classes[flow[i]])
            if child2[i] not in self.service_classes[flow[i]]:
                child2[i] = random.choice(self.service_classes[flow[i]])

        return child1, child2

    def mutate(self, chromosome, flow_idx):
        """Мутація хромосоми."""
        flow = self.flows[flow_idx]
        for i in range(len(chromosome)):
            if random.uniform(0, 1) < self.mutation_prob:
                # Заміняємо сервіс лише з того самого класу
                chromosome[i] = random.choice(self.service_classes[flow[i]])
        return chromosome
    
    def evolve(self, flow_idx):
        """Еволюція популяції протягом кількох поколінь."""
        self.initialize_population(flow_idx)

        for generation in range(self.generations):
            new_population = []

            # Створюємо нову популяцію
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(flow_idx)  # Передаємо flow_idx
                child1, child2 = self.crossover(parent1, parent2, flow_idx)
                child1 = self.mutate(child1, flow_idx)
                child2 = self.mutate(child2, flow_idx)
                new_population.append(child1)
                new_population.append(child2)

            self.population = new_population[:self.population_size]

        # Знаходимо найкращу хромосому (рішення)
        best_solution = max(self.population, key=lambda chromo: self.fitness(chromo, flow_idx))
        best_fitness = self.fitness(best_solution, flow_idx)
        return best_solution, best_fitness
    
class DynamicProgrammingWSC:
    def __init__(self, service_classes, flows, service_data):
        self.service_classes = service_classes  # Класи веб-сервісів
        self.flows = flows  # Набори можливих флоу
        self.service_data = service_data  # Дані веб-сервісів (QoS параметри)
        self.dp_table = {}  # Таблиця для зберігання оптимальних рішень
    
    def get_reward(self, service_class, service_name):
        """Отримує винагороду для конкретного веб-сервісу на основі явно заданих значень QoS."""
        service_idx = self.service_classes[service_class].index(service_name)
        service = self.service_data[service_class - 1][service_idx]
        availability = service['availability']
        exec_time = service['exec_time']
        throughput = service['throughput']

        reward = (availability * 0.4) - (exec_time * 0.4) + (throughput * 0.2)
        return reward

    def find_optimal_service(self, flow_idx, step_idx):
        """Знаходить оптимальний веб-сервіс для поточного кроку флоу."""
        if (flow_idx, step_idx) in self.dp_table:
            return self.dp_table[(flow_idx, step_idx)]
        
        flow = self.flows[flow_idx]
        service_class = flow[step_idx]
        best_service = None
        best_reward = float('-inf')
        
        for service in self.service_classes[service_class]:
            current_reward = self.get_reward(service_class, service)
            
            if step_idx > 0:
                # Додаємо винагороду з попереднього кроку
                prev_service, prev_reward = self.find_optimal_service(flow_idx, step_idx - 1)
                current_reward += prev_reward
            
            if current_reward > best_reward:
                best_reward = current_reward
                best_service = service
        
        # Зберігаємо рішення в таблиці DP
        self.dp_table[(flow_idx, step_idx)] = (best_service, best_reward)
        return best_service, best_reward

    def generate_composite_service(self, flow_idx):
        """Генерує оптимальний композитний веб-сервіс для конкретного флоу."""
        flow_length = len(self.flows[flow_idx])
        optimal_flow = []
        
        for step_idx in range(flow_length):
            service, reward = self.find_optimal_service(flow_idx, step_idx)
            optimal_flow.append(service)
        
        return optimal_flow

    def get_optimal_flow_and_reward(self, flow_idx):
        """Повертає оптимальний флоу і загальну винагороду для поточного флоу."""
        optimal_flow = self.generate_composite_service(flow_idx)
        _, total_reward = self.find_optimal_service(flow_idx, len(self.flows[flow_idx]) - 1)
        return optimal_flow, total_reward

def load_service_data(file_path):
    """Завантаження даних веб-сервісів і флоу з CSV файлів"""
    service_classes = {}
    service_data = []
    flows = []

    # Завантаження сервісів
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_id = int(row['class_id'])
            if class_id not in service_classes:
                service_classes[class_id] = []
                service_data.append([])  # Новий клас
            service_classes[class_id].append(row['service_name'])
            service_data[class_id - 1].append({
                'availability': float(row['availability']),
                'exec_time': float(row['exec_time']),
                'throughput': float(row['throughput']),
            })
    return service_classes, service_data

def load_flows(file_path):
    """Завантаження флоу з CSV файлу"""
    flows = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sequence = list(map(int, row['sequence'].split(';')))
            flows.append(sequence)
    return flows

def train_and_time(algorithm, episodes=500):
    start_time = time.time()
    algorithm.train(episodes)
    end_time = time.time()
    return end_time - start_time

def print_flow_with_qos(flow, service_data, service_classes):
    """Виводить обраний флоу з деталізацією QoS (availability, exec_time, throughput) для кожного сервісу."""
    print("Обраний флоу:")
    for service in flow:
        for class_id, services in service_classes.items():
            if service in services:
                service_idx = services.index(service)
                qos = service_data[class_id - 1][service_idx]
                print(f"Сервіс: {service}, Клас: {class_id}, QoS: "
                      f"Availability: {qos['availability']}, Exec Time: {qos['exec_time']}, Throughput: {qos['throughput']}")


def calculate_total_reward(flow, service_data, service_classes):
    """Обчислює загальну винагороду для обраного флоу на основі QoS параметрів."""
    total_reward = 0
    for service in flow:
        for class_id, services in service_classes.items():
            if service in services:
                service_idx = services.index(service)
                qos = service_data[class_id - 1][service_idx]
                # Обчислюємо винагороду для сервісу
                reward = (qos['availability'] * 0.4) - (qos['exec_time'] * 0.4) + (qos['throughput'] * 0.2)
                total_reward += reward
    return total_reward

def print_flow_results(flow_idx, flow, total_reward):
    """Виводить результати у форматі: флоу N: [сервіси], винагорода."""
    #print(f"Флоу {flow_idx + 1}: {flow}, Винагорода: {total_reward:.3f}")

def find_optimal_flow(algorithm, service_data, service_classes):
    """Знаходить оптимальний флоу для конкретного алгоритму і повертає його з винагородою."""
    best_flow = None
    best_reward = float('-inf')
    
    for flow_idx in range(len(flows)):
        flow = algorithm.generate_composite_service(flow_idx)
        total_reward = calculate_total_reward(flow, service_data, service_classes)
        if total_reward > best_reward:
            best_flow = flow
            best_reward = total_reward
            
    return best_flow, best_reward

def print_optimal_flow_results(algorithm_name, flow_idx, flow, reward):
    """Виводить результати у форматі: Оптимальний флоу для <алгоритму>: Флоу N: [сервіси], винагорода."""
    print(f"Оптимальний флоу для {algorithm_name}: Флоу {flow_idx + 1} {flow}, Винагорода: {reward:.3f}")

# Завантаження даних
service_classes, service_data = load_service_data('services_100.csv')
flows = load_flows('flows_100.csv')

# Q-Learning
q_learning = QLearningWSC(service_classes, flows, service_data)
time_q = train_and_time(q_learning)

print("\nРезультати для Q-Learning:")
best_q_flow, best_q_reward = None, float('-inf')
best_q_flow_idx = None
for flow_idx in range(len(flows)):
    q_learning_flow = q_learning.generate_composite_service(flow_idx)
    total_reward_q = calculate_total_reward(q_learning_flow, service_data, service_classes)
    print_flow_results(flow_idx, q_learning_flow, total_reward_q)

    # Зберігаємо оптимальний флоу
    if total_reward_q > best_q_reward:
        best_q_flow = q_learning_flow
        best_q_reward = total_reward_q
        best_q_flow_idx = flow_idx

# Виведення оптимального флоу для Q-Learning
print_optimal_flow_results('Q-Learning', best_q_flow_idx, best_q_flow, best_q_reward)
print(f"Час виконання для Q-Learning: {time_q:.5f} секунд")

# SARSA
sarsa = SARSALearningWSC(service_classes, flows, service_data)
time_sarsa = train_and_time(sarsa)

print("\nРезультати для SARSA:")
best_sarsa_flow, best_sarsa_reward = None, float('-inf')
best_sarsa_flow_idx = None
for flow_idx in range(len(flows)):
    sarsa_flow = sarsa.generate_composite_service(flow_idx)
    total_reward_sarsa = calculate_total_reward(sarsa_flow, service_data, service_classes)
    print_flow_results(flow_idx, sarsa_flow, total_reward_sarsa)

    # Зберігаємо оптимальний флоу
    if total_reward_sarsa > best_sarsa_reward:
        best_sarsa_flow = sarsa_flow
        best_sarsa_reward = total_reward_sarsa
        best_sarsa_flow_idx = flow_idx

# Виведення оптимального флоу для SARSA
print_optimal_flow_results('SARSA', best_sarsa_flow_idx, best_sarsa_flow, best_sarsa_reward)
print(f"Час виконання для SARSA: {time_sarsa:.5f} секунд")

# Value Iteration
#value_iteration = ValueIterationWSC(service_classes, flows, service_data)
#start_time = time.time()
#value_iteration.value_iteration(iterations=500)
#time_value_iteration = time.time() - start_time

#print("\nРезультати для Value Iteration:")
#best_vi_flow, best_vi_reward = None, float('-inf')
#best_vi_flow_idx = None
#for flow_idx in range(len(flows)):
#    vi_flow = value_iteration.generate_composite_service(flow_idx)
#    total_reward_vi = calculate_total_reward(vi_flow, service_data, service_classes)
#    print_flow_results(flow_idx, vi_flow, total_reward_vi)

    # Зберігаємо оптимальний флоу
#    if total_reward_vi > best_vi_reward:
#        best_vi_flow = vi_flow
#        best_vi_reward = total_reward_vi
#        best_vi_flow_idx = flow_idx

# Виведення оптимального флоу для Value Iteration
#print_optimal_flow_results('Value Iteration', best_vi_flow_idx, best_vi_flow, best_vi_reward)


# Жадібний алгоритм
greedy_wsc = GreedyWSC(service_classes, flows, service_data)
start_time = time.time()
all_greedy_flows, all_greedy_rewards = greedy_wsc.find_flows_and_rewards()
greedy_flow, greedy_reward = greedy_wsc.find_optimal_flow()
time_greedy = time.time() - start_time

print("\nРезультати для Жадібного алгоритму:")
best_greedy_flow_idx = None
for idx, (flow, reward) in enumerate(zip(all_greedy_flows, all_greedy_rewards)):
    print_flow_results(idx, flow, reward)

    if reward == greedy_reward:
        best_greedy_flow_idx = idx

# Виведення оптимального флоу для Жадібного алгоритму
print_optimal_flow_results('Жадібний алгоритм', best_greedy_flow_idx, greedy_flow, greedy_reward)
print(f"Час виконання для Greedy: {time_greedy:.5f} секунд")

# Генетичний алгоритм
#ga_wsc = GeneticAlgorithmWSC(service_classes, flows, service_data)
#start_time = time.time()
#best_ga_flows = []
#best_ga_rewards = []
#for flow_idx in range(len(flows)):
#    best_flow, best_fitness = ga_wsc.evolve(flow_idx)
#    best_ga_flows.append(best_flow)
#    best_ga_rewards.append(best_fitness)
#time_ga = time.time() - start_time

# Виведення результатів для GA
#print("\nРезультати для Генетичного алгоритму:")
#best_ga_flow_idx = None
#best_ga_flow, best_ga_reward = None, float('-inf')
#for idx, (flow, reward) in enumerate(zip(best_ga_flows, best_ga_rewards)):
#    print_flow_results(idx, flow, reward)

#    if reward > best_ga_reward:
#        best_ga_flow = flow
#        best_ga_reward = reward
#        best_ga_flow_idx = idx

# Виведення оптимального флоу для Генетичного алгоритму
#print_optimal_flow_results('Генетичний алгоритм', best_ga_flow_idx, best_ga_flow, best_ga_reward)
#print(f"Час виконання для Greedy: {time_ga:.5f} секунд")


# Динамічне програмування
dp_wsc = DynamicProgrammingWSC(service_classes, flows, service_data)
start_time = time.time()
best_dp_flows = []
best_dp_rewards = []
for flow_idx in range(len(flows)):
    best_flow, best_reward = dp_wsc.get_optimal_flow_and_reward(flow_idx)
    best_dp_flows.append(best_flow)
    best_dp_rewards.append(best_reward)
time_dp = time.time() - start_time

# Виведення результатів для DP
print("\nРезультати для Динамічного програмування:")
best_dp_flow_idx = None
best_dp_flow, best_dp_reward = None, float('-inf')
for idx, (flow, reward) in enumerate(zip(best_dp_flows, best_dp_rewards)):
    print_flow_results(idx, flow, reward)

    if reward > best_dp_reward:
        best_dp_flow = flow
        best_dp_reward = reward
        best_dp_flow_idx = idx

# Виведення оптимального флоу для Динамічного програмування
print_optimal_flow_results('Динамічне програмування', best_dp_flow_idx, best_dp_flow, best_dp_reward)
print(f"Час виконання для Динамічного програмування: {time_dp:.5f} секунд")

# Порівняння часу виконання
#algorithms = ['Q-Learning', 'SARSA', 'Value Iteration', 'Greedy', 'Genetic Algorithm', 'Dynamic Programming']
#times = [time_q, time_sarsa, time_value_iteration, time_greedy, time_ga, time_dp]
#algorithms = ['Q-Learning', 'SARSA', 'Greedy', 'Genetic', 'Dynamic']
#times = [time_q, time_sarsa, time_greedy, time_ga, time_dp]
algorithms = ['Q-Learning', 'SARSA', 'Greedy', 'Dynamic']
times = [time_q, time_sarsa, time_greedy, time_dp]

# Створення графіку
fig, ax = plt.subplots()
bars = ax.bar(algorithms, times, color=['blue', 'green', 'red', 'purple', 'orange'])

# Додавання числових значень над стовпцями
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 5), ha='center', va='bottom')

#plt.bar(algorithms, times)
plt.xlabel('Алгоритми')
plt.ylabel('Час виконання (секунди)')
plt.title('Порівняння часу навчання і виконання алгоритмів')
plt.show()

# Порівняння винагород для кожного алгоритму
optimal_rewards = [
    best_q_reward,
    best_sarsa_reward,
#    best_vi_reward,
    greedy_reward,
#    0, #max(best_ga_rewards),
    max(best_dp_rewards)
]

# Створення графіку
fig, ax = plt.subplots()
bars = ax.bar(algorithms, optimal_rewards, color=['blue', 'green', 'red', 'purple', 'orange'])

# Додавання числових значень над стовпцями
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 5), ha='center', va='bottom')

#plt.bar(algorithms, optimal_rewards)
plt.xlabel('Алгоритми')
plt.ylabel('Сумарна винагорода')
plt.title('Порівняння винагород алгоритмів')
plt.show()
