# Quantum Machine Learning with Quantum Embedding Kernels

This project demonstrates the development of a quantum machine learning model using Quantum Embedding Kernels (QEKs) for a binary classification task. The project is implemented in a single Jupyter Notebook file.

## Overview

Quantum machine learning leverages the principles of quantum computing to enhance traditional machine learning algorithms. This project focuses on using Quantum Embedding Kernels to perform binary classification on a custom dataset. The key components include:

- Generation of a custom dataset with distinct sectors.
- Definition of a quantum kernel using parameterized quantum circuits.
- Training of a Support Vector Machine (SVM) classifier using the quantum kernel.
- Optimization of the kernel parameters to improve classification accuracy.
- Implementation of a parameterized quantum circuit for embedding data into quantum states (quantum feature map).
- Computation of the QEK matrix for the given dataset.
- Optimization of the variational parameters of the quantum feature map by maximizing the kernel-target alignment.
- Training of a support vector machine (SVM) classifier using the optimized QEK matrix.
- Evaluation of the classification accuracy of the SVM classifier.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- `numpy`
- `pennylane`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using pip:

```bash
pip install numpy pennylane matplotlib scikit-learn
```

## Usage
- Clone this repository to your local machine.
- Navigate to the directory containing the cloned repository.
- Open the Jupyter Notebook file (Quantum_Machine_Learning_QEK.ipynb) using Jupyter Notebook or Jupyter Lab.
- Run the notebook cells sequentially to execute the code.

## Code Description
 - **Dataset Generation**
The code includes functions to generate a custom dataset with distinct sectors for the binary classification task:
```
def _make_new_sector_data(num_sectors):
    # Generate sector data with specific angles and colors
    # Returns x, y coordinates, labels, and colors
```
 - **Quantum Kernel Definition**
Quantum circuits are defined to create the quantum kernel:
```
def layer(x, params, wires, i0=0, inc=1):
    # Define a single layer of the quantum circuit

def custom_ansatz(x, params, wires):
    # Define the custom ansatz for embedding the data

def custom_kernel(x1, x2, params):
    # Define the quantum kernel function\
```
 - **Training and Evaluation**
The kernel is used to train an SVM classifier. The kernel parameters are optimized to improve alignment with the target:
svm = SVC(kernel=init_kernel).fit(X, Y)
```
# Train the SVM classifier using the initial kernel

opt = qml.GradientDescentOptimizer(0.2)
# Optimize the kernel parameters

svm_trained = SVC(kernel=trained_kernel_matrix).fit(X, Y)
# Train the SVM classifier using the trained kernel
```
```
from sklearn.ensemble import RandomForestClassifier

# Define the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X, Y)

```
```
from sklearn.neighbors import KNeighborsClassifier

# Define the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn_classifier.fit(X, Y)

```
 - **Results**
The accuracy of the classifier is evaluated before and after the optimization of the kernel parameters.

## Conclusion
This project illustrates the application of Quantum Embedding Kernels in machine learning. The optimization of quantum kernel parameters can significantly enhance the performance of classical machine learning algorithms like SVM.

Feel free to explore the code and modify it to experiment with different datasets and quantum circuit designs.

##### GEET, IIT ROORKEE
