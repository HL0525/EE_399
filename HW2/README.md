# Project Title

This project involves exploring the Yale Faces dataset, which consists of grayscale images of human faces. The goal of the project is to perform various data analysis tasks on this dataset, including computing and visualizing correlation matrices, performing singular value decomposition (SVD), and computing the percentage of variance captured by each SVD mode.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)

## Introduction

The code first loads the Yale Faces dataset using the loadmat() function from the scipy.io library. The dataset contains a matrix X that consists of 2414 columns (i.e., images) and 1024 rows (i.e., pixels). To reduce the dimensionality of the dataset, the code takes only the first 100 columns of X.

The next step is to compute the correlation matrix of X using the formula C = X.T @ X. The code then visualizes this correlation matrix using the pcolor() function from the matplotlib.pyplot library. The plot shows the correlation between each pair of images.

The code then finds the indices of the maximum and minimum correlation values in the correlation matrix using the argmax() and argmin() functions. It extracts the corresponding columns from X and plots them as images using the imshow() function.

Next, the code selects a subset of 10 images from X using specified indices and computes the correlation matrix of this subset. It visualizes this correlation matrix using pcolor().

The code then performs SVD on the original X matrix using the eigh() function from the scipy.linalg library. It extracts the first six SVD modes and plots them as images using imshow(). It also computes the percentage of variance captured by each mode and displays this information. Finally, it computes the norm of the difference between the first eigenvector of X and the first column of the principal component matrix and displays this value.

## Theoretical Background

### Linear Algebra:

Correlation matrix: a symmetric matrix that measures the correlation between each pair of variables in a dataset.
Singular Value Decomposition (SVD): a factorization of a matrix into three matrices, which provides a useful way to analyze the properties of the original matrix.

### Image Processing:

Eigenfaces: a set of eigenvectors derived from a dataset of face images, which can be used to represent faces as linear combinations of these vectors.
Principal Component Analysis (PCA): a technique that uses SVD to find the principal components of a dataset and reduce its dimensionality.

In this project, we use the Yale Face Database, which contains images of 15 individuals under various lighting conditions. We first compute the correlation matrix of the dataset and visualize it using a heat map. We then extract the images with the maximum and minimum correlation values and display them.Next, we perform SVD on the entire dataset and extract the first six eigenvectors (i.e., eigenfaces) to represent the variations in the face images. We also calculate the percentage of variance captured by each eigenface.Finally, we apply PCA to a subset of the dataset and compare the first principal component (PC) with the first eigenvector from SVD. We also plot the first six SVD modes to visualize the variations in the face images captured by the eigenfaces.

## Algorithm Implementation and Development 

The first part of the code loads the necessary libraries: NumPy, SciPy, and Matplotlib. It also loads the Yale Faces dataset using the loadmat function from 
SciPy's io module.
```
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.linalg import eigh

results = loadmat('yalefaces.mat')
X = results['X'][:, :100]  # take first 100 columns of X
```
The X variable contains the images of the dataset. The first 100 images are used for the correlation matrix computation.
```
C = np.dot(X.T, X)  # compute correlation matrix

# plot correlation matrix using pcolor
plt.figure(figsize=(8, 8))
plt.pcolor(C)
plt.colorbar()
plt.title('Correlation Matrix for First 100 Images')
plt.xlabel('Image Index')
plt.ylabel('Image Index')
plt.show()
```
The correlation matrix is computed using NumPy's dot function. The pcolor function from Matplotlib is used to plot the matrix.

Next, the maximum and minimum correlation values are found, and the corresponding columns are extracted from X.

```
max_idx = np.unravel_index(np.argmax(C - np.eye(C.shape[0])*np.max(C)), C.shape)
min_idx = np.unravel_index(np.argmin(C + np.eye(C.shape[0])*np.max(C)), C.shape)

# extract corresponding columns from X
max_img1, max_img2 = X[:, max_idx[0]], X[:, max_idx[1]]
min_img1, min_img2 = X[:, min_idx[0]], X[:, min_idx[1]]

# plot the images
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
ax[0, 0].imshow(max_img1.reshape(32, 32), cmap='gray')
ax[0, 0].set_title('Image {}'.format(max_idx[0] + 1))

ax[0, 1].imshow(max_img2.reshape(32, 32), cmap='gray')
ax[0, 1].set_title('Image {}'.format(max_idx[1] + 1))

ax[1, 0].imshow(min_img1.reshape(32, 32), cmap='gray')
ax[1, 0].set_title('Image {}'.format(min_idx[0] + 1))

ax[1, 1].imshow(min_img2.reshape(32, 32), cmap='gray')
ax[1, 1].set_title('Image {}'.format(min_idx[1] + 1))
plt.tight_layout()
plt.show()
```

The unravel_index function is used to find the indices of the maximum and minimum correlation values in the matrix. The corresponding columns are then extracted from X, reshaped to their original size, and plotted using Matplotlib's imshow function.

In the next section of the code, a subset of columns from X is selected, and a correlation matrix is computed.

```
# specify indices
idx = [0, 312, 511, 4, 2399, 112, 1023, 86, 313, 2004]

# extract columns from X
X_sub = X[:, idx]

# compute correlation matrix
C_sub = np.dot(X_sub.T, X_sub)

# plot correlation matrix
plt.pcolor(C_sub)
plt.colorbar()
plt.show()
```
The subset of columns is specified by the idx variable. The columns are extracted using NumPy's array slicing

## Computational Results

### Correlation Matrix of Images
The first part of the code computes the correlation matrix of the first 100 images in the Yale Faces dataset. The correlation matrix is computed as the dot product of the transpose of X with X. The resulting matrix is then plotted using pcolor.
![image](https://user-images.githubusercontent.com/129907047/232376240-9eda4c65-4745-4cd1-b048-8f93f1d2bbf0.png)


### Maximum and Minimum Correlations
The code then finds the indices of the maximum and minimum correlation values in the correlation matrix. The corresponding columns from X are extracted and plotted as images using imshow.

The four images with the highest and lowerst correlation are shown below:

![image](https://user-images.githubusercontent.com/129907047/232376308-849d3d89-2824-42ff-b55e-5a9418bc06f7.png)

### Correlation Matrix of Selected Images
Next, the code selects 10 images from the Yale Faces dataset and computes the correlation matrix for these images. The resulting correlation matrix is plotted using pcolor.

![image](https://user-images.githubusercontent.com/129907047/232376431-d5e1ef61-d54e-40b7-a43e-73f051bd6f9f.png)

SVD Modes
Finally, the code computes the first 6 SVD modes for the entire Yale Faces dataset. The first 6 SVD modes are plotted as images using imshow.

![image](https://user-images.githubusercontent.com/129907047/232376463-404fe933-e3b6-4451-8aa2-5c5b82d5d762.png)


## Summary and Conclusions

In this project, we applied principal component analysis (PCA) to a dataset of face images. PCA is a dimensionality reduction technique that allows us to extract the most important features from high-dimensional data. By applying PCA to the face images, we were able to identify the most important modes of variation among the images.

We started by computing the correlation matrix of the dataset, and then extracting the most correlated and least correlated images. We then applied PCA to the entire dataset to identify the most important modes of variation, and plotted the first 6 modes. Finally, we selected a subset of images and applied PCA to that subset, and compared the first eigenvector from the subset to the first eigenvector from the entire dataset.

Our results showed that PCA was able to successfully identify the most important modes of variation among the face images, and that the first eigenvector from the subset was similar to the first eigenvector from the entire dataset. Overall, this project demonstrates the power and usefulness of PCA as a dimensionality reduction technique, particularly in the context of image analysis.


