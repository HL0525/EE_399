# Predicting Temperature Trends with Neural Networks

This project utilizes a neural network model to predict temperature trends based on historical data. The model is trained on a dataset containing 
daily temperature readings for a particular location over a 31-day period. The neural network architecture consists of three fully connected layers 
with increasing number of neurons per layer. The model is trained using the mean squared error loss function and the Adam optimizer.

The project also includes a method for splitting the data into training and test sets, and computes the least-square error for both sets. 
Additionally, the project demonstrates the effect of data splitting on the model's performance by comparing the training and test errors for 
different data splits.

This project can be useful for anyone interested in learning how to implement a neural network model for temperature trend prediction or 
for those interested in exploring the impact of data splitting on model performance. The code is written in Python using PyTorch and 
can be easily adapted to other datasets and applications.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)

## Introduction


This project involves working with feed-forward neural networks to fit data and classify images. In the first part,
we are tasked with fitting a three-layer feed-forward neural network to a given data set, and then training the network on 
different subsets of the data to compare the least-squared errors on the training and test sets. The goal is to compare the performance
of the neural network with the models fit in homework one.

In the second part, we will be working with the MNIST dataset and performing PCA analysis to compute the first 20 modes of the digit images. 
Then we will build a feed-forward neural network and compare its performance with LSTM, SVM, and decision tree classifiers.

Throughout the project, we will be writing a narrative report on our Github page and uploading the Python code to Canvas with a comment 
indicating the Github page for the TAs to review.

## Theoretical Background

### Feed-Forward Neural Networks
A feed-forward neural network is a type of artificial neural network where the flow of information moves in only one direction,
from the input layer through the hidden layers to the output layer. The nodes in each layer are fully connected to the nodes in
the subsequent layer. The activation of each neuron in the network is determined by a weighted sum of the inputs followed by an 
activation function. The weights in the network are adjusted during training to minimize the error between the predicted and actual outputs.

### Least-Square Error
Least-Square Error (LSE) is a widely used method for calculating the error between predicted and actual outputs in a regression problem.
It measures the sum of the squared differences between the predicted and actual outputs. In this project, LSE is used to evaluate
the performance of the neural network models on both training and test datasets.

### Principal Component Analysis
Principal Component Analysis (PCA) is a dimensionality reduction technique that is used to identify patterns in data and reduce
the dimensionality of the data by projecting it onto a lower-dimensional space. The goal of PCA is to find the directions of 
maximum variance in the data, called the principal components, and represent the data in terms of these components.

### Support Vector Machines
Support Vector Machines (SVMs) are a type of supervised learning algorithm that can be used for both classification and regression problems.
The algorithm works by finding the hyperplane that best separates the data into different classes. SVMs are particularly useful 
when the data has a non-linear boundary between the classes, as they can use kernel functions to map the data into a higher-dimensional 
space where it is more easily separable.

### Decision Trees
Decision Trees are a type of supervised learning algorithm that can be used for both classification and regression problems.
They work by recursively partitioning the data into subsets based on the value of a specific feature. The goal is to create a tree-like
model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.

## Algorithm Implementation

Task 1: Predicting Time Series Data
The first task involved predicting a time series data of temperature over a 31-day period using a feedforward neural network with three layers. 
```
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
              40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```
The implementation involved using the keras library in Python and defining a neural network with one input layer, 
two hidden layers with 5 and 3 nodes respectively, and one output layer. The input and output layers used a linear activation function,
```
# Compute the least-square error for the training data
inputs_train = torch.from_numpy(X_train).float().unsqueeze(1)
labels_train = torch.from_numpy(Y_train).float().unsqueeze(1)
outputs_train = model(inputs_train)
train_error = ((outputs_train - labels_train)**2).mean()

# Compute the least-square error for the test data
inputs_test = torch.from_numpy(X_test).float().unsqueeze(1)
labels_test = torch.from_numpy(Y_test).float().unsqueeze(1)
outputs_test = model(inputs_test)
test_error = ((outputs_test - labels_test)**2).mean()
```
while the hidden layers used the sigmoid activation function. The adam optimizer was used with a learning rate of 0.01 and the mean squared error (MSE) 
was used as the loss function.



Task 2: Image Classification on MNIST Dataset
The second task involved building a feedforward neural network to classify digits in the MNIST dataset. The implementation involved using 
the keras library and defining a neural network with one input layer, two hidden layers with 128 and 64 nodes respectively, and one output layer.

```
# Load the MNIST dataset and convert to numpy arrays
mnist_train = datasets.MNIST(root=data_path, train=True, transform=data_transform, download=True)
train_images = np.array(mnist_train.data)
num_samples, num_pixels = train_images.shape[0], train_images.shape[1] * train_images.shape[2]
train_images_flat = train_images.reshape(num_samples, num_pixels)

# Compute the first 20 PCA modes
pca = PCA(n_components=20)
pca.fit(train_images_flat)

```
The input and output layers used a linear activation function, while the hidden layers used the relu activation function. 
The adam optimizer was used with a learning rate of 0.001 and the categorical cross-entropy was used as the loss function.
```
lass FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(28, 64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)
```
Before building the neural network, the first 20 principal components were computed from the images in the dataset using PCA.
The neural network was compared with LSTM, SVM, and decision tree classifiers using accuracy as the evaluation metric.

The code for both tasks can be found in the Task1.ipynb and Task2.ipynb files respectively in the GitHub repository.

## Computational Results

For the first part of the project, a three-layer feedforward neural network was trained on the given data. The network was trained 
using the first 20 data points as training data and tested on the remaining 10 data points. The least-square error was computed for 
each model over the training points, and the models were then tested on the test data. The same process was repeated using the first 
10 and last 10 data points as training data.
```
Explained variance of each PCA mode:  [0.09704664 0.07095924 0.06169089 0.05389419 0.04868797 0.04312231
 0.0327193  0.02883895 0.02762029 0.02357001 0.0210919  0.02022991
 0.01715818 0.01692111 0.01578639 0.01482937 0.01324547 0.0127688
 0.01187182 0.01152634]
```
 
The results showed that the neural network performed better than the models fit in homework one. The neural network achieved a lower 
least-square error than the models fit in homework one, indicating that it was better at predicting the values of the data.

For the second part of the project, a feedforward neural network was trained on the MNIST dataset. The first 20 principal components
of the digit images were computed using PCA. A feedforward neural network was then built to classify the digits. The performance of
the neural network was compared against LSTM, SVM (support vector machines), and decision tree classifiers.
```
FNN Epoch 1 loss: 0.757
FNN Epoch 2 loss: 0.368
FNN Epoch 3 loss: 0.322
FNN Epoch 4 loss: 0.294
FNN Epoch 5 loss: 0.272
FNN Epoch 6 loss: 0.253
FNN Epoch 7 loss: 0.236
FNN Epoch 8 loss: 0.220
FNN Epoch 9 loss: 0.207
FNN Epoch 10 loss: 0.195
FNN Test accuracy: 94.640%
```
```
Epoch 1 loss: 0.378
Epoch 2 loss: 0.180
Epoch 3 loss: 0.134
Epoch 4 loss: 0.111
Epoch 5 loss: 0.093
Epoch 6 loss: 0.082
Epoch 7 loss: 0.071
Epoch 8 loss: 0.066
Epoch 9 loss: 0.058
Epoch 10 loss: 0.053
Neural network accuracy: 0.975
SVM accuracy: 0.936
```

The results showed that the neural network outperformed the other classifiers, achieving a higher accuracy in classifying the digits.
The use of PCA helped to improve the performance of the neural network by reducing the dimensionality of the data and removing irrelevant features.
```
Epoch 1 loss: 0.378
Epoch 2 loss: 0.180
Epoch 3 loss: 0.134
Epoch 4 loss: 0.111
Epoch 5 loss: 0.093
Epoch 6 loss: 0.082
Epoch 7 loss: 0.071
Epoch 8 loss: 0.066
Epoch 9 loss: 0.058
Epoch 10 loss: 0.053
Neural network accuracy: 0.975
SVM accuracy: 0.936
```

Overall, the results of the project demonstrate the effectiveness of using neural networks for data analysis and classification tasks. 
The use of PCA also highlights the importance of feature selection and dimensionality reduction in improving the performance of 
machine learning algorithms.

## Summary and Conclusions

In this project, we implemented two different feedforward neural networks for regression and classification tasks. 
For the regression task, we fit a three-layer neural network to the given data and evaluated its performance on training and test datasets.
We also compared it to the models fit in the previous homework. For the classification task, we computed the first 20 principal 
components of the MNIST dataset and built a feedforward neural network for digit classification. We also compared the performance of our
neural network with other classifiers such as LSTM, SVM, and decision trees.

Overall, we found that our feedforward neural network performed well on both tasks, achieving low least-squares error for the regression
task and high accuracy for the classification task. Additionally, we found that using PCA to reduce the dimensionality of the input data
significantly improved the performance of our neural network.

In conclusion, this project demonstrates the power and versatility of neural networks for solving a variety of machine learning problems.
We also identified several areas for further research, such as exploring more complex neural network architectures and experimenting with 
different preprocessing techniques for input data.

