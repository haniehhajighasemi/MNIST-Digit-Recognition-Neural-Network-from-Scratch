import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'weight': weight, 'bias': bias})

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_propagation(self, X):
        activations = [X]
        current_input = X

        for i in range(len(self.layers) - 1):
            z = np.dot(current_input, self.layers[i]['weight']) + self.layers[i]['bias']
            current_input = self.relu(z)
            activations.append(current_input)

        z_output = np.dot(current_input, self.layers[-1]['weight']) + self.layers[-1]['bias']
        output = self.softmax(z_output)
        activations.append(output)

        return activations

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward_propagation(self, X, y_true, activations):
        batch_size = X.shape[0]
        gradients = []
        dA = activations[-1] - y_true

        for i in reversed(range(len(self.layers))):
            dW = np.dot(activations[i].T, dA) / batch_size
            db = np.mean(dA, axis=0, keepdims=True)
            gradients.insert(0, {'dW': dW, 'db': db})

            if i > 0:
                dA_prev = np.dot(dA, self.layers[i]['weight'].T)
                dA = dA_prev * self.relu_derivative(activations[i])

        return gradients

    def update_parameters(self, gradients):
        for i in range(len(self.layers)):
            self.layers[i]['weight'] -= self.learning_rate * gradients[i]['dW']
            self.layers[i]['bias'] -= self.learning_rate * gradients[i]['db']

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]

                activations = self.forward_propagation(batch_X)
                gradients = self.backward_propagation(batch_X, batch_y, activations)
                self.update_parameters(gradients)

            if epoch % 10 == 0:
                train_pred = self.forward_propagation(X_train)[-1]
                val_pred = self.forward_propagation(X_val)[-1]

                train_loss = self.compute_loss(y_train, train_pred)
                val_loss = self.compute_loss(y_val, val_pred)
                train_acc = self.accuracy(y_train, train_pred)
                val_acc = self.accuracy(y_val, val_pred)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)

                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        predictions = self.forward_propagation(X)[-1]
        return np.argmax(predictions, axis=1)

    def accuracy(self, y_true, y_pred):
        predicted_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        return np.mean(predicted_classes == true_classes)