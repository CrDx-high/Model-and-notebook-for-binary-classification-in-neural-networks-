# Model-and-notebook-for-binary-classification-in-neural-networks-
#### Neural Network Binary Classification with PyTorch

## Project Report

This project is a hands-on implementation of **binary classification using neural networks in PyTorch**. The notebook builds a complete classification workflow from scratch using a synthetic non-linear dataset generated with Scikit-Learn's `make_circles()` function.

The main purpose of this project is to understand how neural networks behave on binary classification problems, especially when the data is not linearly separable. The project starts with simple linear models, observes their limitations, and then improves the architecture by introducing non-linear activation functions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Loss Function and Optimizer](#loss-function-and-optimizer)
- [Prediction Pipeline](#prediction-pipeline)
- [Model Experiments](#model-experiments)
- [Decision Boundary Visualization](#decision-boundary-visualization)
- [Key Learnings](#key-learnings)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Conclusion](#conclusion)

---

## Project Overview

Binary classification is one of the most common machine learning tasks. The goal is to classify data into one of two possible classes.

In this project, the model is trained to classify points into two circular classes. Since the data is arranged in circles, a simple linear model cannot separate it properly. This makes the dataset useful for learning why neural networks need **non-linear activation functions**.

The notebook covers the full PyTorch pipeline:

1. Creating data
2. Visualizing data
3. Converting data into tensors
4. Splitting data into training and testing sets
5. Building neural network models
6. Training models
7. Evaluating models
8. Plotting decision boundaries
9. Improving model performance using non-linearity

---

## Problem Statement

The problem is to build a neural network that can correctly classify two classes of circular data.

The model receives two input features:

```text
X1, X2
```

The model predicts one binary label:

```text
0 or 1
```

The challenge is that the data is **non-linear**. A straight line cannot separate the two classes properly. Therefore, the model needs to learn a curved decision boundary.

---

## Dataset

The dataset is created using Scikit-Learn:

```python
from sklearn.datasets import make_circles

n_samples = 1000
X, y = make_circles(n_samples=n_samples,
                    noise=0.03,
                    random_state=42)
```

### Dataset Details

| Property | Value |
|---|---:|
| Dataset type | Synthetic toy dataset |
| Function used | `make_circles()` |
| Total samples | 1000 |
| Input features | 2 |
| Output classes | 2 |
| Noise | 0.03 |
| Random state | 42 |

The dataset contains two circular patterns. Each data point has two coordinates and one class label.

Example input shape:

```text
X shape: (1000, 2)
y shape: (1000,)
```

This means there are 1000 data points, and each point has two features.

---

## Data Preparation

The original data is generated as NumPy arrays. Since PyTorch works with tensors, the data is converted into PyTorch tensors:

```python
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
```

The dataset is then split into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Train-Test Split

| Split | Samples | Percentage |
|---|---:|---:|
| Training data | 800 | 80% |
| Testing data | 200 | 20% |

---

## Project Workflow

The project follows a standard PyTorch machine learning workflow:

```text
Create data
↓
Visualize data
↓
Convert data to tensors
↓
Split into train and test sets
↓
Build model
↓
Choose loss function and optimizer
↓
Train model
↓
Evaluate model
↓
Visualize decision boundary
↓
Improve model architecture
```

---

## Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Building and training neural networks |
| Scikit-Learn | Dataset creation and train-test split |
| Matplotlib | Data and decision boundary visualization |
| Pandas | Displaying dataset in tabular format |
| NumPy | Numerical array handling |
| Requests | Downloading helper functions |
| Jupyter Notebook | Interactive development environment |

---

## Model Architecture

The project experiments with multiple neural network architectures.

---

## Model 0: Basic Linear Neural Network

The first model uses two linear layers.

```python
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
```

### Architecture

```text
Input layer: 2 features
Hidden layer: 5 units
Output layer: 1 unit
```

### Observation

This model does not perform well because it only uses linear layers. The dataset is circular, so the model cannot create a curved decision boundary.

---

## Model 1: Deeper Linear Neural Network

The second model increases the number of hidden units and adds an extra layer.

```python
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
```

### Architecture

```text
Input layer: 2 features
Hidden layer 1: 10 units
Hidden layer 2: 10 units
Output layer: 1 unit
```

### Changes Made

- Increased hidden units from 5 to 10
- Added one more layer
- Increased training epochs from 100 to 1000

### Observation

Even after adding more layers and training for longer, the model still struggles. This is because stacking only linear layers still behaves like a linear transformation.

More linear layers without activation functions do not solve the non-linear nature of the problem.

---

## Model 2: Neural Network with Non-Linearity

The final model adds ReLU activation functions between the linear layers.

```python
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
```

### Architecture

```text
Input layer: 2 features
Hidden layer 1: 10 units
ReLU activation
Hidden layer 2: 10 units
ReLU activation
Output layer: 1 unit
```

### Why ReLU Matters

ReLU is a non-linear activation function. It turns negative values into zero and keeps positive values unchanged.

```text
ReLU(x) = max(0, x)
```

This gives the neural network the ability to learn non-linear patterns.

### Observation

After adding ReLU, the model becomes capable of learning a curved decision boundary. This is the main breakthrough in the project.

---

## Training Process

The training loop follows the standard PyTorch process:

1. Set model to training mode
2. Perform forward pass
3. Convert logits to prediction labels
4. Calculate loss
5. Calculate accuracy
6. Reset optimizer gradients
7. Perform backpropagation
8. Update model parameters
9. Evaluate on test data

Core training code:

```python
model.train()

# Forward pass
y_logits = model(X_train).squeeze()
y_pred = torch.round(torch.sigmoid(y_logits))

# Calculate loss and accuracy
loss = loss_fn(y_logits, y_train)
acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

# Optimizer zero grad
optimizer.zero_grad()

# Backpropagation
loss.backward()

# Gradient descent
optimizer.step()
```

Testing loop:

```python
model.eval()

with torch.inference_mode():
    test_logits = model(X_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))

    test_loss = loss_fn(test_logits, y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
```

---

## Loss Function and Optimizer

### Loss Function

The project uses:

```python
loss_fn = nn.BCEWithLogitsLoss()
```

`BCEWithLogitsLoss` is used for binary classification problems.

It combines:

```text
Sigmoid activation + Binary Cross Entropy loss
```

This means the model should output **raw logits** during training. The sigmoid function is handled internally by the loss function.

### Optimizer

The optimizer used is Stochastic Gradient Descent:

```python
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
```

### Learning Rate

The learning rate used is:

```text
0.1
```

The learning rate controls how large the model's parameter updates are during training.

If the learning rate is too high, the model can overshoot and become unstable. If it is too low, the model may learn too slowly.

---

## Prediction Pipeline

The model outputs raw logits. These logits are not directly class labels.

To convert logits into final predictions:

```python
y_pred = torch.round(torch.sigmoid(y_logits))
```

The full prediction flow is:

```text
Raw logits → Sigmoid probabilities → Rounded class labels
```

### Explanation

| Stage | Meaning |
|---|---|
| Logits | Raw model outputs |
| Sigmoid probabilities | Values between 0 and 1 |
| Rounded labels | Final class predictions, either 0 or 1 |

Classification rule:

```text
If probability >= 0.5 → class 1
If probability < 0.5 → class 0
```

---

## Accuracy Function

A custom accuracy function is used to evaluate model performance:

```python
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
```

This function compares the predicted labels with the actual labels and returns the accuracy percentage.

---

## Model Experiments

### Experiment 1: Simple Linear Model

The first model used two linear layers.

Result:

- Model failed to learn the circular pattern properly
- Decision boundary remained mostly linear
- Accuracy did not improve meaningfully

Reason:

- The dataset is non-linear
- Linear layers alone cannot model circular separation

---

### Experiment 2: Deeper Linear Model

The second model added:

- More hidden units
- More layers
- More epochs

Result:

- Performance still did not improve enough
- The model still lacked non-linearity

Reason:

- Multiple linear layers without activation functions are still mathematically equivalent to a linear transformation

---

### Experiment 3: Non-Linear Model with ReLU

The final model added ReLU activation functions between linear layers.

Result:

- Model performance improved significantly
- Decision boundary became curved
- The model became capable of learning the circular classification pattern

Reason:

- ReLU introduced non-linearity
- The model could now learn complex boundaries

---

## Decision Boundary Visualization

Decision boundaries are plotted using helper functions from the Learn PyTorch repository:

```python
from helper_functions import plot_decision_boundary, plot_predictions
```

The helper file is downloaded with:

```python
import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("helper_functions.py exists, skipping download")
else:
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
    )
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)
```

Decision boundary plotting code:

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
```

This visualization is important because it shows whether the model is actually learning the shape of the data.

---

## Key Learnings

### 1. Shape matters in PyTorch

The input data has shape:

```text
(1000, 2)
```

The model must be designed to accept 2 input features.

```python
nn.Linear(in_features=2, out_features=10)
```

---

### 2. Binary classification needs the correct loss function

For binary classification, `BCEWithLogitsLoss` is a strong choice because it expects raw logits and internally applies sigmoid.

---

### 3. Logits are not probabilities

Raw model outputs must be converted before interpreting them as predictions.

```text
logits → sigmoid → probabilities → labels
```

---

### 4. More layers do not always mean better performance

Adding more linear layers did not solve the problem because the dataset itself was non-linear.

---

### 5. Non-linearity is essential

The main issue was not the number of layers. The missing piece was a non-linear activation function.

ReLU allowed the model to learn a curved decision boundary.

---

### 6. Visualization helps debug models

The decision boundary plot clearly showed whether the model was learning or not. This made it easier to understand why the first models failed.

---

## Project Structure

```text
.
├── Neural_network_classification_pytorch.ipynb
├── helper_functions.py
├── images/
│   └── Ml_pipeline.png
└── README.md
```

### File Description

| File | Description |
|---|---|
| `Neural_network_classification_pytorch.ipynb` | Main notebook containing the complete project |
| `helper_functions.py` | Helper functions for plotting decision boundaries |
| `images/Ml_pipeline.png` | Machine learning pipeline image used in the notebook |
| `README.md` | Project report and documentation |

---

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Move Into the Project Folder

```bash
cd <your-project-folder>
```

### 3. Install Dependencies

```bash
pip install torch scikit-learn matplotlib pandas numpy requests jupyter
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the Notebook

Open:

```text
Neural_network_classification_pytorch.ipynb
```

Then run the cells from top to bottom.

---

## Requirements

The project requires the following Python libraries:

```text
torch
scikit-learn
matplotlib
pandas
numpy
requests
jupyter
```

Install them using:

```bash
pip install torch scikit-learn matplotlib pandas numpy requests jupyter
```

---

## Common Issues and Fixes

### Import Error with `make_circles`

Correct import:

```python
from sklearn.datasets import make_circles
```

Incorrect spelling like `make_cricles` will cause an import error.

---

### Matplotlib Import Error

If `matplotlib.pyplot` fails to import, possible fixes include:

```bash
pip uninstall matplotlib -y
pip install matplotlib
```

Also make sure there is no local file named:

```text
matplotlib.py
```

because that can conflict with the actual Matplotlib package.

---

### Shape Mismatch

For binary classification, make sure model outputs and labels have compatible shapes.

The model output is often squeezed:

```python
y_logits = model(X_train).squeeze()
```

This changes the output shape from:

```text
[800, 1]
```

to:

```text
[800]
```

which matches the shape of `y_train`.

---

## Conclusion

This project demonstrates the complete process of building a binary classification neural network in PyTorch.

The most important learning from this project is that **model architecture must match the nature of the data**. For circular non-linear data, simply adding more linear layers is not enough. The model needs non-linear activation functions such as ReLU to learn complex decision boundaries.

Through this notebook, the project covers the essential foundations of PyTorch classification:

- Tensor conversion
- Train-test splitting
- Neural network construction
- Loss functions
- Optimizers
- Training loops
- Evaluation loops
- Accuracy calculation
- Logits and sigmoid conversion
- Decision boundary visualization
- Model improvement through non-linearity

This project is a solid foundational step toward understanding how neural networks solve classification problems and how model design decisions directly affect learning performance.

---

## Final Reflection

This notebook helped build a practical understanding of binary classification beyond just writing code. It showed why neural networks need the right architecture, why visualization matters, and why debugging model behavior is as important as training the model itself.

The major takeaway is simple:

> A neural network does not become powerful just because it has more layers. It becomes powerful when the architecture is designed correctly for the problem.
