import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from numpy.typing import ArrayLike

### Glove word embeddings
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

def sentence_to_glove_vector(sentence, word_to_vec, embedding_dim=300):
    """
    Converts a sentence to a vector by averaging word embeddings.

    Args:
        sentence (str): Input sentence
        word_to_vec (dict): Dictionary mapping word -> GloVe vector
        embedding_dim (int): Size of each word vector

    Returns:
        np.ndarray: Vector representation of the sentence (shape = (embedding_dim,))
    """
    words = sentence.lower().split()
    vectors = []

    for word in words:
        if word in word_to_vec:
            vectors.append(word_to_vec[word])
        else:
            # If the word is unknown, we can use a zero vector
            vectors.append(np.zeros((embedding_dim,)))

    if len(vectors) == 0:
        # In case sentence has no known words
        return np.zeros((embedding_dim,))
    else:
        return np.mean(vectors, axis=0)

### Load dataset
def load_imdb_dataset(path='IMDB Dataset.csv', max_features=5000):
    global fitted_vectorizer
    df = pd.read_csv(path)

    texts = df['review'].astype(str).tolist()
    sentiments = df['sentiment'].astype(str).tolist()

    # Encode sentiment (positive = 1, negative = 0)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(sentiments)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=True,
        stop_words='english',
        strip_accents='unicode',
        token_pattern=r'\b\w+\b'
    )

    X = vectorizer.fit_transform(texts).toarray()
    y = labels.reshape(1, -1)

    fitted_vectorizer = vectorizer
    return X.T, y, label_encoder

### He parameters initialization
def he_initialization(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return parameters

### Regularization methods
def l2_regularization(parameters, lambd, m):
    l2_cost = sum(np.sum(np.square(parameters[key])) for key in parameters if "W" in key)
    return (lambd / (2 * m)) * l2_cost

def apply_dropout(A, keep_prob):
    D = np.random.rand(*A.shape) < keep_prob
    A = np.multiply(A, D)
    A /= keep_prob
    return A, D

def update_learning_rate(initial_lr, decay_rate, epoch):
    return initial_lr / (1 + decay_rate * epoch)

### Opimizer algorithms
class Optimizers:
    def __init__(self, parameters):
        self.v = {"d" + key: np.zeros_like(val) for key, val in parameters.items()}
        self.s = {"d" + key: np.zeros_like(val) for key, val in parameters.items()}

    def gradient_descent(self, grads, parameters, lr):
        for key in parameters:
            parameters[key] -= lr * grads["d" + key]
        return parameters

    def momentum(self, grads, parameters, lr, beta=0.9):
        for key in parameters:
            self.v["d" + key] = beta * self.v["d" + key] + (1 - beta) * grads["d" + key]
            parameters[key] -= lr * self.v["d" + key]
        return parameters

    def rmsprop(self, grads, parameters, lr, beta=0.999, epsilon=1e-8):
        for key in parameters:
            self.s["d" + key] = beta * self.s["d" + key] + (1 - beta) * np.square(grads["d" + key])
            parameters[key] -= lr * grads["d" + key] / (np.sqrt(self.s["d" + key]) + epsilon)
        return parameters

    def adam(self, grads, parameters, lr, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
        for key in parameters:
            self.v["d" + key] = beta1 * self.v["d" + key] + (1 - beta1) * grads["d" + key]
            self.s["d" + key] = beta2 * self.s["d" + key] + (1 - beta2) * np.square(grads["d" + key])
            v_corrected = self.v["d" + key] / (1 - beta1 ** t)
            s_corrected = self.s["d" + key] / (1 - beta2 ** t)
            parameters[key] -= lr * v_corrected / (np.sqrt(s_corrected) + epsilon)
        return parameters

### Activation functions and derivatives
def relu(Z: ArrayLike) -> ArrayLike:
    return np.maximum(0, Z)

def relu_derivative(Z: ArrayLike) -> ArrayLike:
    return (Z > 0).astype(float)

def sigmoid(Z: ArrayLike) -> ArrayLike:
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z: ArrayLike) -> ArrayLike:
    s = sigmoid(Z)
    return s * (1 - s)

def tanh(Z: ArrayLike) -> ArrayLike:
    return np.tanh(Z)

def tanh_derivative(Z: ArrayLike) -> ArrayLike:
    return 1 - np.tanh(Z)**2

def softmax(Z: ArrayLike) -> ArrayLike:
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_stable)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot_encode(y, num_classes):
    m = y.shape[1]
    one_hot = np.zeros((num_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot