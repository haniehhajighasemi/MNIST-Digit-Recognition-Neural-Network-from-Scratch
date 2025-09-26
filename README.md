**MNIST Digit Recognition – Neural Network from Scratch**
=========================================================

This project implements a **fully connected feedforward neural network (multilayer perceptron)** from scratch in **NumPy** to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

Unlike frameworks such as TensorFlow or PyTorch, all key steps — forward propagation, backpropagation, weight updates, and evaluation — are implemented manually. This makes it a great learning resource to understand how neural networks work under the hood.

**✨ Features**
--------------

*   Loads and preprocesses the MNIST dataset (28×28 grayscale digits).
    
*   Normalizes pixel values to \[0, 1\].
    
*   One-hot encodes labels (0–9).
    
*   Implements:
    
    *   Dense layers with **ReLU** activation.
        
    *   **Softmax** output for multiclass classification.
        
    *   **Cross-entropy loss**.
        
    *   **Backpropagation** and gradient descent.
        
*   Trains using **mini-batch gradient descent**.
    
*   Visualizes sample digits and training history.
    
*   Evaluates accuracy and loss on test set.
    

**🏗️ Network Architecture**
----------------------------

The default architecture is:
```
Input: 784 neurons (28x28 pixels)
↓
Hidden Layer 1: 128 neurons, ReLU activation
↓
Hidden Layer 2: 64 neurons, ReLU activation
↓
Output Layer: 10 neurons, Softmax activation
```
**📂 Project Structure**
------------------------
```
project/
├── utils/               # Python package folder
│   ├── __init__.py        # Makes this a package
│   ├── neural_network.py  # NeuralNetwork class
│   ├── data_utils.py      # Data loading & preprocessing
│   ├── viz_utils.py       # Visualization
├── main.py                # Entry point
└── README.md              # Documentation             # Project documentation
```
**🚀 Usage**
------------

### **1\. Install dependencies**
```
pip install numpy matplotlib scikit-learn pandas
```
### **2\. Run the training pipeline**
```
python main.py
```
### **3\. Expected output**

*   Training progress printed every 10 epochs:
    
```
Epoch 0: Train Loss: 2.302, Val Loss: 2.301, Train Acc: 0.112, Val Acc: 0.110
...
Epoch 90: Train Loss: 0.085, Val Loss: 0.120, Train Acc: 0.976, Val Acc: 0.969
```
Visualization of:

*   Sample digits with labels.
    
*   Training/validation loss and accuracy curves.
    

**📊 Results**
--------------

*   Achieves around **96–97% test accuracy** after training for 100 epochs.
    
*   Demonstrates correct predictions on sample test digits.
    

**🔍 Key Insights**
-------------------

*   How forward propagation computes activations.
    
*   How backpropagation flows gradients through layers.
    
*   Why activation functions like **ReLU** are important.
    
*   How **Softmax + Cross-Entropy** simplify gradient computation.
    

**📝 To-Do / Extensions**
-------------------------

*   Add support for different architectures (configurable hidden layers).
    
*   Implement **momentum** or **Adam optimizer**.
    
*   Try with a **Convolutional Neural Network (CNN)** for higher accuracy.
    
*   Add model saving/loading.
    

**📖 References**
-----------------

*   [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
    
*   Neural Networks and Deep Learning – Michael Nielsen
    
*   Deep Learning Specialization (Coursera, Andrew Ng)