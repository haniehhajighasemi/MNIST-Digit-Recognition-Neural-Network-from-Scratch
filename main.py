import time
import numpy as np

from project import NeuralNetwork, load_and_preprocess_mnist, visualize_samples, plot_training_history

def main():
    print("MNIST Neural Network Implementation")
    print("=" * 40)

    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_mnist()

    # Visualize some samples
    print("\nSample digits from the dataset:")
    visualize_samples(X_train, y_train)

    # Create neural network
    network = NeuralNetwork(input_size=784, hidden_sizes=[128, 64], output_size=10, learning_rate=0.01)

    print("\nNeural Network Architecture:")
    print("Input Layer: 784 neurons (28x28 pixels)")
    print("Hidden Layer 1: 128 neurons (ReLU activation)")
    print("Hidden Layer 2: 64 neurons (ReLU activation)")
    print("Output Layer: 10 neurons (Softmax activation)")

    # Train
    print("\nStarting training...")
    start_time = time.time()
    train_losses, val_losses, train_accs, val_accs = network.train(
        X_train, y_train, X_val, y_val, epochs=100, batch_size=64
    )
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # Evaluate
    test_predictions = network.forward_propagation(X_test)[-1]
    test_accuracy = network.accuracy(y_test, test_predictions)
    test_loss = network.compute_loss(y_test, test_predictions)

    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")

    # Sample predictions
    print("\nSample predictions:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    sample_predictions = network.predict(X_test[sample_indices])
    sample_true = np.argmax(y_test[sample_indices], axis=1)

    for i, idx in enumerate(sample_indices):
        print(f"Sample {i+1}: Predicted = {sample_predictions[i]}, "
              f"Actual = {sample_true[i]}, "
              f"Correct = {'✓' if sample_predictions[i] == sample_true[i] else '✗'}")

if __name__ == "__main__":
    main()