import math


class Neuron:
    def __init__(self):
        self.weights = [-0.8, 0.1]

    def activationFunction(self, pa):
        # Sigmoid activation function
        print("Activation function input:", pa)
        return 1 / (1 + math.exp(-pa))

    def output(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Input length must match weights length")

        pa = 0

        for w, i in zip(self.weights, inputs):
            pa += w * i

        return self.activationFunction(pa)

    def train(self, X, y):

        for epoch in range(100):

            epoch_error = 0

            print(f"Epoch {epoch + 1}")

            for i in range(len(X)):
                inputs = X[i]
                target = y[i]

                predicted = self.output(inputs)
                error = target - predicted

                epoch_error += abs(error)

                print(
                    f"Inputs: {inputs}, Predicted: {predicted}, Target: {target}, Error: {error}"
                )

                learning_rate = 0.01

                # Update weights based on error
                for i in range(len(self.weights)):
                    self.weights[i] += learning_rate * error * inputs[i]

            if epoch_error == 0:
                print("Training complete with no errors.")
                break



def main():
    X = [[1, 1], [1, 0], [0, 1], [0, 0]]

    y = [1, 0, 0, 0]

    neuron = Neuron()
    neuron.train(X, y)
    print("Final weights:", neuron.weights)

    predictions = [neuron.output(inputs) for inputs in X]
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
