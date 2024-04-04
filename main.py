import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier Initialization
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        # Softmax activation function
        expZ = np.exp(z - np.max(z))
        return expZ / expZ.sum(axis=1, keepdims=True)

    def forward_propagation(self, X):
        # Implementing Forward Propagation
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward_propagation(self, X, y, y_hat):
        # Implementing Backward Propagation
        m = X.shape[0]
        dZ2 = y_hat - y
        dW2 = 1 / m * np.dot(self.A1.T, dZ2)
        db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.W2.T) * (self.A1 * (1 - self.A1))  # derivative of the sigmoid function
        dW1 = 1 / m * np.dot(X.T, dZ1)
        db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)

        # store the gradients
        self.grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_weights(self, learning_rate):
        # Updating weights
        self.W1 = self.W1 - learning_rate * self.grads["dW1"]
        self.b1 = self.b1 - learning_rate * self.grads["db1"]
        self.W2 = self.W2 - learning_rate * self.grads["dW2"]
        self.b2 = self.b2 - learning_rate * self.grads["db2"]

    def fit(self, X, y, epochs, batch_size, learning_rate):
        # Training the model

        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation, :]
            y_shuffled = y[permutation, :]
            for i in range(0, X.shape[0], batch_size):
                i += 1
                X_batch = X_shuffled[i:i + batch_size, :]
                y_batch = y_shuffled[i:i + batch_size, :]
                y_hat = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_batch, y_hat)
                self.update_weights(learning_rate)



    def predict(self, X):
        # Making a prediction
        y_hat = self.forward_propagation(X)
        return np.argmax(y_hat, axis=1)

    def accuracy(self, X, y):
        # Making a prediction
        y_hat = self.forward_propagation(X)
        predictions = np.argmax(y_hat, axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)


def grid_search():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = (train_X/255).astype('float32')
    test_X = (test_X/255).astype('float32')
    train_y=  np.eye(10)[train_y.astype('int')]
    test_y = np.eye(10)[test_y.astype('int')]
    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    # define the grid
    grid = {
        'hidden_size': [50, 100, 200],
        'epochs': [10, 20, 30],
        'batch_size': [32, 64, 128],
        'learning_rate': [0.1, 0.01, 0.001],
    }

    # create a list of all possible combinations of parameters
    from itertools import product
    parameter_combinations = list(product(*grid.values()))

    # loop through the parameter combinations
    best_score = 0
    best_params = None
    i=0
    for params in parameter_combinations:
        i+=1
        print("started iter", i)
        hidden_size, epochs, batch_size, learning_rate = params

        # initialize a new neural network
        nn = NeuralNetwork(input_size=784, hidden_size=hidden_size, output_size=10)

        # fit the neural network
        nn.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

        # get the score on the validation data
        score = nn.accuracy(val_X, val_y)

        # if this score is better than the previous best, update the best score and best parameters
        if score > best_score:
            best_score = score
            best_params = params

    print("Best validation accuracy: ", best_score)
    print("Best parameters: ", best_params)

def Test():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = (train_X / 255).astype('float32')
    test_X = (test_X / 255).astype('float32')
    train_y = np.eye(10)[train_y.astype('int')]
    test_y = np.eye(10)[test_y.astype('int')]
    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)
    nn = NeuralNetwork(784,200,10)
    nn.fit(train_X,train_y,30,32,0.1)
    score = nn.accuracy(test_X, test_y)
    print(score)





if __name__ == "__main__":
    Test()