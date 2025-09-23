# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Load the Iris dataset using a suitable library.

### STEP 2: 

Preprocess the data by handling missing values and normalizing features.

### STEP 3: 

Split the dataset into training and testing sets.

### STEP 4: 

Train a classification model using the training data.


### STEP 5: 

Evaluate the model on the test data and calculate accuracy.

### STEP 6: 

Display the test accuracy, confusion matrix, and classification report.

## PROGRAM

### Name: SANJAY C

### Register Number: 212223240150

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
```
```py
# Load Iris dataset
iris = load_iris()
print(iris)
X = iris.data  # Features
y = iris.target  # Labels (already numerical)
```
```py
# Convert to DataFrame for easy inspection
df = pd.DataFrame(X, columns=iris.feature_names)
print(df)
df['target'] = y
print(df)
```
```py
# Display first and last 5 rows
print("First 5 rows of dataset:\n", df.head())
print("\nLast 5 rows of dataset:\n", df.tail())
```
```py
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```py
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```py
# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
```
```py
# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
```
```py
# Define Neural Network Model
class IrisClassifier(nn.Module):
    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)
```
```py
# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
     for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
```
```py
# Initialize model, loss function, and optimizer
model = IrisClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```py
# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=100)
```
```py
# Evaluate the model
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())
```
```py
# Compute metrics
accuracy = accuracy_score(actuals, predictions)
conf_matrix = confusion_matrix(actuals, predictions)
class_report = classification_report(actuals, predictions, target_names=iris.target_names)

# Print details

print("\nName:SANJAY C")
print("Register No: 212223240150")
print(f'Test Accuracy: {accuracy:.2f}%')
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
```
```py
# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```
```py
# Make a sample prediction
sample_input = X_test[5].unsqueeze(0)  # Removed unnecessary .clone()
with torch.no_grad():
    output = model(sample_input)
    predicted_class_index = torch.argmax(output[0]).item()
    predicted_class_label = iris.target_names[predicted_class_index]

print("\nName:SANJAY C")
print("Register No: 212223240150")
print(f'Predicted class for sample input: {predicted_class_label}')
print(f'Actual class for sample input: {iris.target_names[y_test[5].item()]}')
```

### Dataset Information
<img width="849" height="562" alt="image" src="https://github.com/user-attachments/assets/1d87ec88-2e7c-4898-8408-3ebe25bfcba1" />


### OUTPUT

## Confusion Matrix

<img width="648" height="585" alt="image" src="https://github.com/user-attachments/assets/35a229c6-a3f2-4a0e-9fe0-192db8266ea1" />


## Classification Report
<img width="628" height="385" alt="image" src="https://github.com/user-attachments/assets/77cda67e-059d-457c-acb2-6e88a0e52641" />



### New Sample Data Prediction
<img width="633" height="118" alt="image" src="https://github.com/user-attachments/assets/165d897b-e26f-4d27-85e6-11605a9db01f" />


## RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch.
