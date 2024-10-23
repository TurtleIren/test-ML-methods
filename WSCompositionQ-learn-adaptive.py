import numpy as np
import random

class QLearningWSC:
    def __init__(self, service_classes, flows, service_data, gamma=0.8, alpha=0.1, epsilon=0.2):
        self.service_classes = service_classes  # Класи веб-сервісів (з іменами сервісів)
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
            for step in range(len(flow) + 1):
                states.append(f"flow_{i}_state_{step}")
        return states

    def generate_actions(self):
        """Генерує дії - вибір сервісу з кожного класу для кожного флоу."""
        actions = {}
        for i, flow in enumerate(self.flows):
            for step, service_class in enumerate(flow):
                # Вибираємо імена сервісів відповідного класу
                actions[f"flow_{i}_state_{step}"] = self.service_classes[service_class]
        return actions

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
            self.q_table[state] = {action: 0 for action in self.actions[state]}
        
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
                    reward = self.get_reward(action)
                    next_state = f"flow_{flow_idx}_state_{step + 1}"

                    # Оновлення Q-таблиці
                    self.update_q_table(state, action, reward, next_state)

                    state = next_state
                    if state == f"flow_{flow_idx}_state_{len(flow)}":
                        break  # Кінцевий стан досягнуто

    def generate_composite_service(self, flow_idx):
        """Генерує композитний веб-сервіс для конкретного флоу на основі навченої політики."""
        state = f"flow_{flow_idx}_state_0"
        flow = []
        for step in range(len(self.flows[flow_idx])):
            action = self.choose_action(state)
            flow.append(action)
            state = f"flow_{flow_idx}_state_{step + 1}"
        return flow

    def find_optimal_flow(self):
        """Знаходить оптимальний композитний сервіс серед флоу на основі суми винагород."""
        best_flow = None
        best_reward_sum = float('-inf')

        for flow_idx, flow in enumerate(self.flows):
            total_reward = 0
            state = f"flow_{flow_idx}_state_0"
            for step in range(len(flow)):
                action = self.choose_action(state)
                reward = self.get_reward(action)
                total_reward += reward
                state = f"flow_{flow_idx}_state_{step + 1}"
            
            # Якщо сума винагород для поточного флоу є кращою, оновлюємо найкращий флоу
            if total_reward > best_reward_sum:
                best_reward_sum = total_reward
                best_flow = flow_idx

        return best_flow, best_reward_sum


# Фіксовані значення для веб-сервісів у кожному класі
service_data = [
    # Клас 1
    [
        {'availability': 0.95, 'exec_time': 0.2, 'throughput': 50},
        {'availability': 0.9, 'exec_time': 0.3, 'throughput': 40},
    ],
    # Клас 2
    [
        {'availability': 0.98, 'exec_time': 0.1, 'throughput': 70},
        {'availability': 0.9, 'exec_time': 0.25, 'throughput': 55},
        {'availability': 0.88, 'exec_time': 0.35, 'throughput': 65},
    ],
    # Клас 3
    [
        {'availability': 0.97, 'exec_time': 0.2, 'throughput': 90},
        {'availability': 0.89, 'exec_time': 0.5, 'throughput': 75},
    ],
    # Клас 4
    [
        {'availability': 0.99, 'exec_time': 0.2, 'throughput': 100},
        {'availability': 0.92, 'exec_time': 0.25, 'throughput': 85},
    ],
]

# Опис класів веб-сервісів
service_classes = {
    1: ['сервіс_1', 'сервіс_2'],  # Клас 1
    2: ['сервіс_3', 'сервіс_4', 'сервіс_5'],  # Клас 2
    3: ['сервіс_6', 'сервіс_7'],  # Клас 3
    4: ['сервіс_8', 'сервіс_9'],  # Клас 4
}

# Набори флоу
flows = [
    [1, 2],  # Флоу 1: клас 1, клас 2
    [1, 3, 4],  # Флоу 2: клас 1, клас 3, клас 4
]

# Навчання
q_learning_wsc = QLearningWSC(service_classes, flows, service_data)
q_learning_wsc.train(episodes=500)

# Генерація композитного сервісу для кожного флоу
for flow_idx in range(len(flows)):
    composite_service_flow = q_learning_wsc.generate_composite_service(flow_idx)
    print(f"Згенерований композитний сервіс для флоу {flow_idx + 1}:", composite_service_flow)


# Знайти оптимальний флоу
best_flow, best_reward_sum = q_learning_wsc.find_optimal_flow()
print(f"Найкращий флоу: {best_flow + 1}, із загальною винагородою: {best_reward_sum}")
