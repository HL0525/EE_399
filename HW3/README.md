# EE 399 SPRING QUATER 2023
# Instructor: J. Nathan Kutz
# HOMEWORK #3:
# DUE: Midnight on 4/24 (Extra credit if turned in by 4/21)

This project is an analysis of the popular MNIST dataset, which contains a large set of handwritten digits. The analysis involves applying singular value decomposition (SVD) to the digit images, exploring the singular value spectrum, and determining the rank of the digit space required for good image reconstruction.The U, Σ, and V matrices are also interpreted in this analysis. A 3D plot is created by projecting onto three selected V-modes (columns) and colored by their digit label.

The second part of the analysis involves building linear classifiers, specifically Linear Discriminant Analysis (LDA), to identify individual digits in the training set. Two and three digits are selected and tested for classification accuracy. The most difficult and easiest pairs of digits to separate are identified and quantified using LDA, Support Vector Machines (SVM), and decision tree classifiers.

All results are discussed in detail and include visualizations and images to aid in understanding the analysis.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)

## Introduction

This project analyzes the popular MNIST dataset of handwritten digits using various machine learning techniques. The MNIST dataset consists of 70,000 grayscale images of handwritten digits, each with a resolution of 28x28 pixels. The goal of this project is to identify and classify the digits in the dataset using linear classifiers, support vector machines (SVMs), and decision trees.

The first part of the project involves performing a singular value decomposition (SVD) analysis of the digit images. This analysis involves reshaping each image into a column vector and using these vectors to construct a data matrix. The singular value spectrum of this matrix is analyzed to determine the number of modes necessary for good image reconstruction.

In the second part of the project, linear classifiers (LDA) are built to identify individual digits in the training set. Two and three digits are chosen to build a classifier to identify them. The accuracy of the separation is quantified using the test data. SVM and decision tree classifiers are also used to compare their performance with LDA.

This project includes detailed code, plots, and explanations of the machine learning techniques used to analyze the MNIST dataset.

## Theoretical Background

The MNIST dataset is a collection of handwritten digits commonly used as a benchmark for image classification algorithms. Each image is 28x28 pixels, and there are 60,000 training images and 10,000 test images.

To analyze the MNIST dataset, we perform an SVD analysis of the digit images. We reshape each image into a column vector and create a data matrix where each column is a different image. The singular value spectrum shows us how many modes are necessary for good image reconstruction, i.e., the rank of the digit space. The interpretation of the U, Σ, and V matrices is also important, as they represent the left singular vectors, the singular values, and the right singular vectors, respectively.

We then use the projected data to build a classifier to identify individual digits in the training set. We start by building a linear classifier (LDA) to identify/classify two digits and then expand to three digits. We also identify which two digits in the dataset are most difficult and easiest to separate and quantify the accuracy of the separation with LDA on the test data.

To compare the performance of different classifiers, we use SVM (support vector machines) and decision tree classifiers, which were the state-of-the-art until about 2014. We then compare the performance of LDA, SVM, and decision trees on the hardest and easiest pair of digits to separate.

## Algorithm Implementation

Singular Value Decomposition (SVD) Analysis:
To perform an SVD analysis of the digit images, each image is reshaped into a column vector, and each column of the data matrix is a different image. The SVD analysis provides three matrices, U, Σ, and V, where U and V are orthonormal matrices, and Σ is a diagonal matrix with singular values. The singular value spectrum is analyzed to determine the number of modes necessary for good image reconstruction, which corresponds to the rank r of the digit space.
```
X = mnist.data.T
U, s, Vt = np.linalg.svd(X, full_matrices=False)
energy = np.cumsum(s**2) / np.sum(s**2)
plt.plot(energy)
plt.xlabel('Number of Singular Values')
plt.ylabel('Energy Captured')
plt.show()
k = 50  # number of singular values to use
X_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
# Reshape the reconstructed data into images
X_approx = X_approx.T.reshape((-1, 28, 28))
```

Linear Discriminant Analysis (LDA) Classifier:
Two and three digits are selected to build a linear classifier for identification/classification. LDA is used to separate the digits based on their features. The accuracy of the separation is quantified for both training and test sets. The two digits that are the most difficult to separate and the two digits that are the easiest to separate are identified and their separability is quantified.

```
# Select only the digit 0 and 1
X_01 = X[(y == '0') | (y == '1')]
y_01 = y[(y == '0') | (y == '1')]

# Perform PCA on the data
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_01)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_01, test_size=0.2)

# Fit the LDA model to the training data
lda = LDA()
lda.fit(X_train, y_train)
```

Support Vector Machines (SVM) and Decision Tree Classifiers:
SVM and decision tree classifiers are also implemented to separate all ten digits. The performance of these classifiers is compared to that of LDA on the most difficult and easiest pairs of digits to separate.

```

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# Evaluate the SVM classifier on the test set
y_pred_svm = svm_clf.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy of SVM classifier: {:.2f}%".format(acc_svm * 100))

# Train a decision tree classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
```

Projection onto Three Selected V-Modes:
The digit images are projected onto three selected V-modes and colored by their digit label. The resulting projection is displayed in a 3D plot.

```
# Perform SVD on the data
svd = TruncatedSVD(n_components=10)
X_svd = svd.fit_transform(X)

# Select three V-modes
v_mode1 = 2
v_mode2 = 3
v_mode3 = 5

# Project the data onto the three selected V-modes
X_proj = X_svd[:, [v_mode1, v_mode2, v_mode3]]

# Create the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], c=y.astype(int), s=20)

# Set the labels and limits of the plot
ax.set_xlabel(f'V-mode {v_mode1}')
ax.set_ylabel(f'V-mode {v_mode2}')
ax.set_zlabel(f'V-mode {v_mode3}')
```

The code also includes data preprocessing steps such as normalization and train-test splitting, as well as functions for plotting the singular value spectrum, confusion matrix, and classification report.


## Computational Results
### Singular Value Spectrum
We first performed an SVD analysis of the MNIST dataset. The singular value spectrum is shown below:
![image](https://user-images.githubusercontent.com/129907047/233766590-e13be3f3-da08-4566-a48e-f5d6683ab0f6.png)
![image](https://user-images.githubusercontent.com/129907047/233766595-77068e3b-4800-4f12-8479-1a9889fd016c.png)
From the plot, we observe that the singular values decay rapidly, and most of the information is captured by the first few modes. We can see that the first 50 modes capture more than 90% of the total variance in the dataset.

### Image Reconstruction
We reconstructed the MNIST images using different numbers of modes. Some of the reconstructed images are shown below:
![image](https://user-images.githubusercontent.com/129907047/233766613-1baf6074-2be5-4ddc-a075-177332dc61a2.png)
We can see that with as few as 50 modes, we can still get reasonable reconstructions of the MNIST images.

### Interpretation of U, Σ, and V
The U, Σ, and V matrices in the SVD decomposition of the MNIST dataset have the following interpretations:

U: The left singular vectors. Each column of U represents a feature or a pattern that captures some aspect of the variation in the dataset.
Σ: The singular values. They represent the amount of variation captured by each feature in U. The larger the singular value, the more important the corresponding feature.
V: The right singular vectors. Each row of V represents an image in the dataset projected onto the feature space defined by U.
![image](https://user-images.githubusercontent.com/129907047/233767099-6f0b8ba2-046c-4f7e-a185-2fad5e586272.png)

### Digit Classification
We built a linear classifier to identify individual digits in the MNIST training set. We picked two digits (3 and 5) and trained the classifier to identify them. The classifier performs well, with an overall accuracy of 96.14%.The classifier performs well, with an overall accuracy of 94.57%.We then evaluated the classifier on the test set. From the confusion matrices, we can see that digits 3 and 5 are the most difficult to separate, with an accuracy of 92.86%. On the other hand, digits 1 and 7 are the easiest to separate, with an accuracy of 99.19%.SVM classifier outperforms both the LDA and decision tree classifiers, with an overall accuracy of 97.88%.

We also trained SVM and decision tree classifiers on the entire dataset and compared their performance to the LDA classifier.

We also picked three digits (1, 4, and 9) and trained the classifier to identify them
## Summary and Conclusions

In this project, we analyzed the MNIST dataset using SVD analysis and built classifiers to identify individual digits. We found that the singular value spectrum showed a significant drop-off in singular values after the 20th mode, suggesting that the rank of the digit space is around 20. The U, Σ, and V matrices represent the left singular vectors, diagonal matrix of singular values, and right singular vectors, respectively, and provide information about the relationship between the original data and the low-rank approximation.

We used linear discriminant analysis (LDA) to build classifiers for two and three selected digits and found that some pairs of digits are more difficult to separate than others. We also compared the performance of LDA, SVM, and decision tree classifiers on the hardest and easiest pairs of digits to separate and found that LDA had the highest accuracy for the easiest pair, while SVM had the highest accuracy for the hardest pair.

Overall, our analysis demonstrated the effectiveness of SVD and LDA in analyzing and classifying digit images. This project can be extended to other image datasets and can provide insights into dimensionality reduction and classification in machine learning.

