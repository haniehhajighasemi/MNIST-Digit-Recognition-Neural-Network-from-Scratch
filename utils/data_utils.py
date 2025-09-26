import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Put dataset loading and preprocessing
def load_and_preprocess_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=int)

    y_one_hot = np.zeros((len(y), 10))
    y_one_hot[np.arange(len(y)), y] = 1

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y_one_hot[:60000], y_one_hot[60000:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test