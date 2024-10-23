import random
import csv

# Налаштування для генерації даних
max_services_per_class = 100
num_classes = 100  # Кількість класів
max_flow_length = 100  # Максимальна довжина флоу
num_flows = 100  # Кількість флоу

# Генерація QoS для кожного сервісу
def generate_service_qos():
    return {
        'availability': round(random.uniform(0.85, 0.99), 2),
        'exec_time': round(random.uniform(0.1, 0.5), 2),
        'throughput': random.randint(50, 100)
    }

# Створюємо сервіси для кожного класу
service_classes = {}
service_data = []

for class_idx in range(1, num_classes + 1):
    service_classes[class_idx] = []
    class_data = []
    num_services = random.randint(1, max_services_per_class)  # Випадкова кількість сервісів у кожному класі
    for service_idx in range(num_services):
        service_name = f"сервіс_{class_idx}_{service_idx + 1}"
        service_classes[class_idx].append(service_name)
        class_data.append(generate_service_qos())
    service_data.append(class_data)

# Створюємо кілька флоу
flows = []
for _ in range(num_flows):
    flow_length = random.randint(1, max_flow_length)  # Випадкова довжина флоу
    flow = [random.randint(1, num_classes) for _ in range(flow_length)]
    flows.append(flow)

# Зберігаємо ці дані у CSV форматі для подальшого використання
with open('services_100.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['class_id', 'service_name', 'availability', 'exec_time', 'throughput'])
    for class_idx, services in service_classes.items():
        for service_name, service_qos in zip(services, service_data[class_idx - 1]):
            writer.writerow([class_idx, service_name, service_qos['availability'], service_qos['exec_time'], service_qos['throughput']])

with open('flows_100.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['sequence'])
    for flow in flows:
        writer.writerow([';'.join(map(str, flow))])
