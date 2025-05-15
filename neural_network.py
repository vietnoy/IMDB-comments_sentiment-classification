import numpy as np
import functions as F

class Neural_network:
    def __init__(self, num_layers, layer_dims, activation="relu", optimizer="gradient_descent", task="binary",
                 regularization=None, lambd=0.0, dropout_keep_prob=1.0, num_epochs=2000,
                 learning_rate=0.01, decay_rate=0.001, initialization="he"):
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.activation = activation
        self.task = task
        self.optimizer_name = optimizer
        self.regularization = regularization
        self.lambd = lambd
        self.keep_prob = dropout_keep_prob
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.initialization = initialization
        self.params_initialize(layer_dims)
        self.optimizer = F.Optimizers(self.parameters)

    def params_initialize(self, layer_dims):
        self.parameters = {}
        for i in range(1, len(layer_dims)):
            if self.initialization == "default":
                self.parameters[f"W{i}"] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
            else:
                self.parameters[f"W{i}"] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2. / layer_dims[i - 1])
            self.parameters[f"b{i}"] = np.zeros((layer_dims[i], 1))

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        if self.task == "multiclass":
            cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        else:
            cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / m
        if self.regularization == "l2":
            cost += F.l2_regularization(self.parameters, self.lambd, m)
        return cost

    def forward_prop(self, X):
        A = X
        caches = [(None, None, A)]
        for l in range(1, self.num_layers):
            W, b = self.parameters[f"W{l}"], self.parameters[f"b{l}"]
            Z = np.dot(W, A) + b
            if self.activation == "relu":
                A = F.relu(Z)
            elif self.activation == "tanh":
                A = F.tanh(Z)
            else:
                A = F.sigmoid(Z)
            if self.keep_prob < 1.0:
                A, D = F.apply_dropout(A, self.keep_prob)
            else:
                D = None
            caches.append((Z, D, A))
        W, b = self.parameters[f"W{self.num_layers}"], self.parameters[f"b{self.num_layers}"]
        Z = np.dot(W, A) + b
        if self.task == "multiclass":
            AL = F.softmax(Z)
        else:
            AL = F.sigmoid(Z)
        caches.append((Z, None, AL))
        return AL, caches

    def backward_prop(self, caches, X, Y):
        grads = {}
        m = X.shape[1]
        AL = caches[-1][2]
        dZ = AL - Y

        for l in reversed(range(1, self.num_layers + 1)):
            A_prev = caches[l - 1][2] if l > 1 else X
            W = self.parameters[f"W{l}"]
            grads[f"dW{l}"] = (1 / m) * np.dot(dZ, A_prev.T)
            if self.regularization == "l2":
                grads[f"dW{l}"] += (self.lambd / m) * W
            grads[f"db{l}"] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dA_prev = np.dot(W.T, dZ)
                Z_prev = caches[l - 1][0]
                if self.activation == "relu":
                    dZ = dA_prev * F.relu_derivative(Z_prev)
                elif self.activation == "tanh":
                    dZ = dA_prev * F.tanh_derivative(Z_prev)
                else:
                    dZ = dA_prev * F.sigmoid_derivative(Z_prev)
        return grads

    def update_parameters(self, grads, epoch):
        lr = F.update_learning_rate(self.learning_rate, self.decay_rate, epoch)
        if self.optimizer_name == "gradient_descent":
            self.parameters = self.optimizer.gradient_descent(grads, self.parameters, lr)
        elif self.optimizer_name == "momentum":
            self.parameters = self.optimizer.momentum(grads, self.parameters, lr)
        elif self.optimizer_name == "rmsprop":
            self.parameters = self.optimizer.rmsprop(grads, self.parameters, lr)
        elif self.optimizer_name == "adam":
            self.parameters = self.optimizer.adam(grads, self.parameters, lr, t=epoch + 1)

    def predict(self, X):
        AL, _ = self.forward_prop(X)
        if self.task == "multiclass":
            return np.argmax(AL, axis=0)
        else:
            return (AL > 0.5).astype(int)

    def fit(self, X, Y, track_loss=False, log_every=100):
        loss_history = []
        for epoch in range(self.num_epochs):
            AL, caches = self.forward_prop(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_prop(caches, X, Y)
            self.update_parameters(grads, epoch)
            if track_loss and (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs} - Cost: {cost:.4f}")
                loss_history.append(cost)
        return loss_history if track_loss else None
