import numpy as np
import math
import matplotlib.pyplot as plt

# Клас нейрону
class Neuron:
    def __init__(self, input_size):
        self.size = input_size                      # Кількість вхідних сигналів нейрону
        self.layer = None                           # Шар, до якого належить екземпляр нейрону
        self.inputs = None                          # Вхідні сигнали нейрону
        self.inputs_grad = None                     # Локальні градієнти вхідних сигналів
        self.inputs_global_grads = None             # Глобальні градієнти вхідних сигналів
        self.output = None                          # Вихідний сигнал нейрону
        self.weights = np.random.uniform(-1, 1, self.size)
        self.weights_grads = None                   # Локальні градієнти ваг нейрону
        self.global_weights_grads = None            # Глобальні градієнти ваг
        self.bias = 0                               # Біас нейрону
        self.bias_grad = 1                          # Локальний градієнт біаса
        self.global_bias_grad = None                # Глобальний градієнт біаса
        self.global_output_grad = 0.0                 # Глобальний градієнт вихідного сигналу
        self.local_output_grad = None               # Локальний градієнт вихідного сигналу
        self.next = list()                          # Список наступних нейронів за зв'язками
        self.prev = list()                          # Список попередніх нейронів за зв'язками
        self.weights_grads_sum = np.array([0.0 for _ in range(self.size)])               # Суми глобальних градієнтів ваг
        self.bias_grads_sum = 0.0                     # Сума глобальних градієнтів біаса

    # Метод введення вхідних сигналів
    def input_signals(self, signals: np.array):
        self.inputs = signals                                       # Вхідні сигнали нейрону
        value = np.dot(self.inputs, self.weights) + self.bias       # Скалярний добуток ваг та сигналів плюс біас
        self.output = self.tanh(value)                              # Значення виходу нейрону після ктивації
        self.local_output_grad = self.tanh_grad(value)              # Локальний градієнт виходу нейрону
        self.weights_grads = np.array(self.inputs)                            # Локальні градієнти ваг нейрону
        self.inputs_grad = self.weights                             # Локальні градієнти вхідних сигналів

    # Метод активації за функцією tanh
    @staticmethod
    def tanh(signal):
        return ((math.exp(signal) - math.exp(-signal))
                / (math.exp(signal) + math.exp(-signal)))

    # Метод обчислення градієнта
    def tanh_grad(self, signal):
        return 1 - self.tanh(signal) ** 2

    # Повернення рядкового значення нейрону
    def __str__(self):
        return f"Neuron signal: {self.output}"

    # Метод оновлення ваг та біаса в процесі навчання
    def learn(self, learning_rate: float, learning_type='batch'):

        self.weights = self.weights - learning_rate * self.global_weights_grads     # Оновлення ваг
        self.bias = self.bias - learning_rate * self.global_bias_grad               # Оновлення біаса


# Клас шару нейронів
class Layer:
    def __init__(self, neurons_amount: int, neurons_size: int):
        self.neurons_amount = neurons_amount                    # Кількість нейронів в шарі
        self.neurons_size = neurons_size                        # Кількість вхідних сигналів нейронів
        self.neurons = list()                                   # Список нейронів шару
        self.prev = None                                        # Посилання на попередній шар
        self.next = None                                        # Посилання на наступний шар

        # Заповнення шару нейронами
        for _ in range(self.neurons_amount):
            neuron = Neuron(self.neurons_size)                  # Створення екземпляру нейрону
            neuron.layer = self                                 # Позначення поточного шару як шару нейрона
            self.neurons.append(neuron)                         # Додавання нейрону до списку нейронів шару

    # Метод введення вхідних сигналів
    def input_signals(self, signals):
        outputs = list()                                        # Список вихідних сигналів шару
        for neuron in self.neurons:
            neuron.input_signals(signals)                       # Введення сигнілів кожному нейрону в шарі
            outputs.append(neuron.output)                       # Додавання вихідного сигналу нейрону до списку

        self.outputs = outputs                                  # Кінцевий список вихідних сигналів шару

    # Повернення рядкового значення нейрону
    def __str__(self):
        return f"Layer: \n {[str(neuron) for neuron in self.neurons]}"

# Клас нейронної мережі
class Network:
    def __init__(self, input_size):
        self.input_size = input_size                            # Кількість вхідних сигналів нейромережі
        self.layers = list()                                    # Список шарів нейронів
        self.inputs = None                                      # Список вхідних сигналів
        self.outputs = None                                     # Список вихідних сигналів
        self.average_losses = None

    # Метод заповнення нейромережі шарами та нейронами
    def set_layers(self, neurons_amount: list):
        self.neurons_amount = neurons_amount                                        # Список кількості нейронів в кожному шарі

        # Проходження по кількості шарів нейромережі
        for i in range(len(self.neurons_amount)):
            self.layers.append(Layer(self.neurons_amount[i],                        # Додавання шару нейронів
                                     self.layers[i-1].neurons_amount if i > 0
                                     else self.input_size))
            if i > 0:                                                               # Прив'язка сусідніх шарів один до одного
                self.layers[i].prev = self.layers[i-1]
                self.layers[i-1].next = self.layers[i]

                for first in self.layers[i-1].neurons:                              # Прив'язка сусідніх нейронів один до одного
                    for second in self.layers[i].neurons:
                        first.next.append(second)
                        second.prev.append(first)

    # Рекурсивний метод введення вхідних сигналів
    def input_signals(self, signals: np.array, cur_layer=None):
        if cur_layer is None:                                                       # Якщо починаємо з першого шару
            cur_layer = self.layers[0]

        cur_layer.input_signals(signals)                                            # Введення сигналів в поточний шар

        # Якщо існує наступний шар
        if cur_layer.next:
            self.input_signals(cur_layer.outputs, cur_layer.next)                   # Рекурсивно вводимо виходи поточного шару у входи наступного
        else:
            self.outputs = cur_layer.outputs                                        # Виходи останнього шару вважаємо виходами мережі

        return self.outputs

    # Повернення рядкового виразу нейромережі
    def __str__(self):
        return f"Network: {[str(layer) for layer in self.layers]}"

    # Метод отримання значення середньої квадратичної помилки
    @staticmethod
    def loss(outputs: np.array, targets: np.array):
        print(outputs)
        print(targets)
        return 1/len(outputs) * sum((targets[i] - outputs[i])**2
                                    for i in range(len(outputs)))

    # Метод отримання градієнта квадратичної помилки
    @staticmethod
    def loss_grad(outputs: np.array, targets: np.array):
        return [2* (outputs[i] - targets[i]) / len(outputs)
                 for i in range(len(outputs))]

    # Метод навчання нейромережі методом BATCH
    def learn_batch(self, sets: np.array, targets: np.array, max_epoсh: int, learning_rate: float):
        self.average_losses = dict()                  # Словник для зберігання середньої помилки за кожну епоху навчання
        average_loss = 0.0                         # Поточне значення середньої квадратичної помилки

        # Навчання в рамках максимальної кількості епох
        for epoch in range(max_epoсh):

            # Проходження по кожному наборі навчальних даних
            for k in range(len(sets)):


                # Обнулення глобальних градієнтів вихідних значень
                for layer in self.layers:
                    for neuron in layer.neurons:
                        neuron.global_output_grad = 0.0

                self.input_signals(np.array(sets[k]))                             # Введення поточних вхідних значень в мережу
                print(self.outputs)
                average_loss += self.loss(self.outputs, targets[k])       # Сума помилок за кожен набір навчальних даних
                loss_grads = self.loss_grad(self.outputs, targets[k])     # Значення градієнта помилки за кожен набір даних

                # Проходження по кожному останньому нейрону мережі
                for i in range(len(loss_grads)):
                    # Встановлення глобального градієнта кожного останнього нейрону
                    self.layers[-1].neurons[i].global_output_grad = self.layers[-1].neurons[i].local_output_grad * loss_grads[i]

                # Зворотнє проходження по всім шарам нейромережі
                for layer in reversed(self.layers):
                    # Проходження по кожному нейрону шару
                    for neuron in layer.neurons:
                        neuron.weights_global_grads = neuron.weights_grads * neuron.global_output_grad       # Обчислення глобальних градієнтів ваг
                        neuron.global_bias_grad = neuron.bias_grad * neuron.global_output_grad              # Обчислення глобального градієнта біаса
                        neuron.weights_grads_sum += neuron.weights_global_grads                              # Обчислення суми всіх глобальних градієнтів ваг
                        neuron.bias_grads_sum += neuron.global_bias_grad                                    # Обчислення суми всіх глобальних градієнтів біаса

                        # Проходження по кожному попередньому нейрону
                        for i in range(len(neuron.prev)):
                            # Обчислення глобального градієнта вихідного значення кожного нейрона
                            neuron.prev[i].global_output_grad += (neuron.weights[i]
                                                                  * neuron.prev[i].local_output_grad * neuron.global_output_grad)


            average_loss /= len(sets)                                   # Обчислення середнього значення помилки
            self.average_losses.update({epoch: average_loss})                # Додавання середнього значення помилки до списку

            # Навчання кожного шару
            for layer in self.layers:
                # Навчання кожного нейрону
                for neuron in layer.neurons:
                    neuron.global_weights_grads = neuron.weights_grads_sum / len(sets)      # Визначення середнього значення ваг
                    neuron.weights_grads_sum = np.array([0.0 for _ in range(neuron.size)])    # Обнулення сум градієнтів ваг
                    neuron.global_bias_grads = neuron.bias_grads_sum / len(sets)            # Визначення середнього значення біаса
                    neuron.bias_grads_sum = 0.0                                               # Обнулення суми градієнта біаса
                    neuron.learn(learning_rate)                                             # Виклик методу навчання нейрону


    # Метод навчання нейромережі методом ONLINE
    def learn_online(self, sets: np.array, targets: np.array, max_epoсh: int, learning_rate: float):
        self.average_losses = dict()                  # Словник для зберігання середньої помилки за кожну епоху навчання
        average_loss = 0.0                            # Поточне значення середньої квадратичної помилки

        # Навчання в рамках максимальної кількості епох
        for epoch in range(max_epoсh):

            # Проходження по кожному наборі навчальних даних
            for k in range(len(sets)):

                # Обнулення глобальних градієнтів вихідних значень
                for layer in self.layers:
                    for neuron in layer.neurons:
                        neuron.global_output_grad = 0.0

                self.input_signals(np.array(sets[k]))                             # Введення поточних вхідних значень в мережу

                average_loss += self.loss(self.outputs, targets[k])       # Сума помилок за кожен набір навчальних даних
                loss_grads = self.loss_grad(self.outputs, targets[k])     # Значення градієнта помилки за кожен набір даних

                # Проходження по кожному останньому нейрону мережі
                for i in range(len(loss_grads)):
                    # Встановлення глобального градієнта кожного останнього нейрону
                    self.layers[-1].neurons[i].global_output_grad = self.layers[-1].neurons[i].local_output_grad * loss_grads[i]

                # Зворотнє проходження по всім шарам нейромережі
                for layer in reversed(self.layers):
                    # Проходження по кожному нейрону шару
                    for neuron in layer.neurons:
                        neuron.global_weights_grads = neuron.weights_grads * neuron.global_output_grad       # Обчислення глобальних градієнтів ваг
                        neuron.global_bias_grad = neuron.bias_grad * neuron.global_output_grad              # Обчислення глобального градієнта біаса

                        # Проходження по кожному попередньому нейрону
                        for i in range(len(neuron.prev)):
                            # Обчислення глобального градієнта вихідного значення кожного нейрона
                            neuron.prev[i].global_output_grad += (neuron.weights[i]
                                                                  * neuron.prev[i].local_output_grad * neuron.global_output_grad)

                # Навчання кожного шару
                for layer in self.layers:
                    # Навчання кожного нейрону
                    for neuron in layer.neurons:
                        neuron.learn(learning_rate)  # Виклик методу навчання нейрону

            average_loss /= len(sets)                                        # Обчислення середнього значення помилки
            self.average_losses.update({epoch: average_loss})                # Додавання середнього значення помилки до списку

    def get_round_outputs(self, signals: np.array):
        outputs = self.input_signals(signals)
        round_outputs = list()
        for output in outputs:
            round_outputs.append(1 if output >= 0 else -1)

        return round_outputs

    def check_network(self, sets, targets):
        amount = len(sets)
        trues = 0
        for i in range(amount):
            if self.get_round_outputs(sets[i]) == targets[i]:
                trues+=1

        true_amount = f'{trues} / {amount}'
        print(true_amount)
        accuracy = trues/amount * 100
        print(str(accuracy) + " %")

if __name__ == "__main__":
    network = Network(2)
    network.set_layers([16, 14, 12, 10, 8, 6, 4, 2, 2])
    signals = np.array([1,0])
    sets = np.array([[0,0], [0,1], [0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]])
    targets = [[1,1],[1,1],[1,1],[1,1],[-1,-1],[1,1],[-1,-1],[1,1],[-1,-1],[-1,-1],[1,1],[1,1],[-1,-1],[-1,-1],[-1,-1],[1,1]]


    # targets = [[1],[1],[1],[1],[0],[1],[0],[1],[0],[0],[1],[1],[0],[0],[0],[1]]
    # targets = [[1],[1],[1],[1],[-1],[1],[-1],[1],[-1],[-1],[1],[1],[-1],[-1],[-1],[1]]
    network.learn_online(sets,targets,1000,0.01)

    for data in sets:
        print(data, ' - ', network.get_round_outputs(data))

    network.check_network(sets, targets)

    epochs = list(network.average_losses.keys())
    losses = list(network.average_losses.values())

    plt.plot(epochs, losses, marker='o', label='Average Loss')
    plt.xlabel('Epoch')  # підпис осі X
    plt.ylabel('Loss')  # підпис осі Y
    plt.title('Зміна помилки під час навчання')  # заголовок
    plt.grid(True)  # сітка
    plt.legend()  # легенда
    plt.show()  # показати графік






