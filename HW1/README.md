# EE_399 HOMEWORK #1:
## Student: Hao Lin, Instructor: J. Nathan Kutz, Date 4/10/2023

The project involves using Python to perform data analysis on a given dataset using various models. 
The dataset consists of 31 data points, and the goal is to find the best model that fits the data with the least-squares error.

1. In the first part of the project, a model of the form f(x) = A cos(Bx) + Cx + D is fit to the data using the least-squares method. 
The code finds the minimum error and determines the values of the parameters A, B, C, and D.

2. In the second part, two of the parameters are fixed, and the other two are swept through various values to generate a 2D loss (error) landscape. 
All combinations of two fixed parameters and two swept parameters are considered, and the results are visualized in a grid.

3. In the third part, three models are fit to the first 20 data points - a line, parabola, and 19th-degree polynomial. 
The least-square error is computed for each of these models over the training points, and then the error is computed for each model on the remaining 10 test points.

4. In the fourth part, the first 10 and last 10 data points are used as training data to fit the models, and the results are compared to those of the third part.

Overall, the project involves using Python to explore various models and techniques for fitting data and evaluating the performance of these models.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)




## Introduction

This code performs a least-squares fitting of a model function to a given set of data, and outputs the parameters that minimize the error. 
Specifically, the model function is given by:

f(x) = A cos(Bx) + Cx + D

where A, B, C, and D are parameters to be determined from the data. The fitting is done using the minimize function from the scipy.
optimize module, with the Nelder-Mead method.

This Python code aims to analyze and model a given set of data using various techniques. The data consists of 31 data points, with corresponding values of X and Y. 
The task is to fit a model to the data with the least-squares error. The chosen model is a combination of a cosine function and a linear function,
with four parameters A, B, C, and D.

In part (i) of the problem, the code computes the values of A, B, C, and D that minimize the least-squares error of the model. 
In part (ii), the code generates a 2D loss (error) landscape by fixing two of the parameters and sweeping through the values of the other two. 
The results are visualized using a grid. The number of minima that can be found as the parameters are swept through is also recorded.

In part (iii), the code uses the first 20 data points as training data to fit three different models:
a line, a parabola, and a 19th degree polynomial. The least-squares error of each model is computed over the training points.
Then, the least-square error of these models is computed on the remaining 10 data points, which are the test data.

In part (iv), the code repeats the same process as in part (iii), but this time, it uses the first 10 and last 10 data points as training data, 
and fits the model to the 10 held-out middle data points.

The code is designed to be readable and well-documented, with clear explanations of the methods and algorithms used.
The code is also structured in a modular way, with separate functions for each part of the problem.

## Theoretical Background

The problem presented in this exercise involves finding the best-fit parameters for a given model to a set of data using least-squares error minimization. 
The model used in this case is a combination of a cosine function and a linear function, represented as f(x) = A cos(Bx) + Cx + D. 
The goal is to find the values of the parameters A, B, C, and D that minimize the sum of squared errors between the model predictions and the actual data points.

Least-squares error minimization is a common method used in regression analysis to find the best-fit parameters for a given model. 
It involves minimizing the sum of squared errors between the model predictions and the actual data points. 
In this case, the sum of squared errors is represented by the expression E = 1/n * ∑(f(xj) - yj)^2, where n is the number of data points, 
xj and yj are the jth data point coordinates, and f(xj) is the corresponding predicted value from the model.

To find the best-fit parameters, a numerical optimization algorithm is used to minimize the sum of squared errors. 
In this case, the code will use the minimize() function from the scipy.optimize library to find the minimum error and 
determine the values of the parameters A, B, C, and D that produce the best fit.

In addition to finding the best-fit parameters, the code will also generate a 2D loss landscape by fixing two parameters and sweeping through values
of the other two parameters. The purpose of this is to visualize how changes in the parameters affect the error of the model predictions. 
The results will be visualized using a grid of colored cells, where each cell represents a combination of two fixed parameters and two swept parameters.

Finally, the code will compare the performance of three different models: a linear function, a parabolic function, and a 19th degree polynomial function.
The least-square error of each model will be computed over the training data, and the best performing model will be selected. 
The selected model will then be tested on the remaining data points to evaluate its generalization performance.

## Algorithm Implementation

### Finding Parameters A, B, C, and D
To find the parameters A, B, C, and D that minimize the least-squares error, we can use the scipy.optimize.curve_fit() function, 
which uses non-linear least squares to fit a function to data. We first define the function f(x, A, B, C, D) as:

```
import numpy as np

def f(x, A, B, C, D):
    return A * np.cos(B * x) + C * x + D
```

Next, we can use curve_fit() to find the optimal values of A, B, C, and D. The function takes the function f, the data X and Y,
and an initial guess for the parameters p0 as input, and returns the optimal values of the parameters and the covariance matrix

### Creating a 2D Loss Landscape
To create a 2D loss landscape, we can fix two of the parameters (A and B, for example) and sweep through values of the other 
two parameters (C and D, for example) to generate a grid of errors. We can use the numpy.meshgrid() function to create a grid of values,
and then evaluate the function f() at each point on the grid:

```
# Define the range of values for C and D
C_range = np.linspace(-1, 1, 101)
D_range = np.linspace(-1, 1, 101)

# Create a meshgrid of C and D values
C_values, D_values = np.meshgrid(C_range, D_range)

# Evaluate the function f() at each point on the grid
Z = np.zeros_like(C_values)
for i in range(len(C_range)):
    for j in range(len(D_range)):
        Z[i,j] = np.mean((f(X, A, B, C_range[i], D_range[j]) - Y)**2)
```

We can then use a plotting library like matplotlib to visualize the results as a grid of colored cells using the pcolor() function:

```
import matplotlib.pyplot as plt

# Plot the loss landscape
fig, ax = plt.subplots()
c = ax.pcolor(C_values, D_values, Z, cmap='viridis')
ax.set_xlabel('C')
ax.set_ylabel('D')
fig.colorbar(c, ax=ax)
plt.show()
```
Fitting Models to Training and Test Data
To fit a line, parabola, and 19th degree polynomial to the first 20 data points, we can use the numpy.polyfit() function, 
which fits a polynomial of degree n to data using least-squares regression. We can then compute the least-squares error for each model over the training data,
and use the models to make predictions on the test data:
```
# Define the first 20 data points as training data
X_train = X[:20]
Y_train = Y[:20]

# Define the last 10 data points as test data
X_test = X[20:]
Y_test = Y[20:]

# Fit a line to the training data
p_line = np.polyfit(X_train, Y_train, 1)
Y_line_train = np.polyval(p_line, X_train)
Y_line_test = np.polyval(p_line, X_test)

```


## Computational Results
Part (i)
After implementing the algorithm, the minimum error was found to be 3.054, with the following parameter values:

A = 5.072
B = 0.195
C = 1.548
D = 34.506
Part (ii)
By fixing two of the parameters and sweeping through the other two, a 2D loss landscape was generated.
The results were visualized using pcolor and can be seen in the following figure:
![image](https://user-images.githubusercontent.com/129907047/231021303-9c443aa8-f042-49bd-8edd-c6a7ecbcb89c.png)
![image](https://user-images.githubusercontent.com/129907047/231021325-430bf2ad-b604-427c-9b78-708f7c7c1876.png)
![image](https://user-images.githubusercontent.com/129907047/231021424-85bd02a7-e24d-4e27-931e-212eb84998f5.png)
![image](https://user-images.githubusercontent.com/129907047/231021440-bf9557a9-6974-4e7f-9474-20c71194ff50.png)
![image](https://user-images.githubusercontent.com/129907047/231021455-e29ffaec-b43a-489f-a70a-a8081ccf0b72.png)
![image](https://user-images.githubusercontent.com/129907047/231021466-2d94789e-2423-4b38-9a35-8244769a59e6.png)

2D Loss Landscape

As we can see from the plot, there are multiple local minima.

Part (iii)
The least square errors for the line, parabola, and 19th degree polynomial were found to be:

Line: 14.825 (training), 18.138 (test)
Parabola: 10.551 (training), 12.677 (test)
19th Degree Polynomial: 0.557 (training), 596.268 (test)
![image](https://user-images.githubusercontent.com/129907047/231021491-9c25c7b9-a5ca-4484-90e2-6cdae76de856.png)

Part (iv)
When using the first and last 10 data points as training data and the middle 10 as test data, the least square errors were:

Line: 1.330 (test)
Parabola: 2.503 (test)
19th Degree Polynomial: 1888.118 (test)
Comparing these results to those obtained in Part (iii), we can see that the polynomial fit 
performs worse when only using the middle 10 data points for testing. The line and parabola fits perform similarly in both cases.
![image](https://user-images.githubusercontent.com/129907047/231021631-b6e2d0f8-69bc-4cdd-ad44-5d19589caff4.png)

## Summary and Conclusions

In this project, we explored the use of least-squares error to fit a model to a given dataset. 
We started by fitting a model of the form f(x) = A cos(Bx) + Cx + D to the dataset using the least-squares error approach. 
We then determined the values of the parameters A, B, C, and D that gave us the minimum error.

Next, we fixed two of the parameters and swept through values of the other two parameters to generate a 2D loss landscape. 
We visualized the results using pcolor and found that there were multiple minima as we swept through parameters.

In the third part of the project, we used the first 20 data points as training data and fit a line, parabola, and 19th degree polynomial to the data.
We computed the least-square error for each of these models over the training points and also on the test data which were the remaining 10 data points.
We found that the 19th degree polynomial had the lowest error on the training data, but performed poorly on the test data, suggesting overfitting.
The line and parabola had similar errors on both the training and test data.

Finally, we repeated the third part of the project using the first 10 and last 10 data points as training data and fitting the model to the test data
(the 10 held out middle data points). We compared these results to the previous part and found that the line and parabola performed better on the test data
than the 19th degree polynomial.

In conclusion, this project highlights the importance of evaluating model performance not just on the training data,
but also on test data to avoid overfitting. The use of least-squares error can be an effective approach to fitting models to datasets, 
but it is important to consider the complexity of the model and the number of parameters to avoid overfitting.
