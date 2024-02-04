# SVM Kernels from Scratch

## Overview

This project explores the creation of Support Vector Machine (SVM) kernels from scratch using Python and NumPy. SVMs are powerful machine learning algorithms for classification tasks, particularly in scenarios where a linear boundary is insufficient for high-dimensional data. The implemented kernels include linear, sigmoid, polynomial, and radial basis function (RBF), both for binary and multiclass SVMs.

## Contributors

- **Austin Lackey**
- **Tomy Sabalo Farias**

## Introduction

Support Vector Machines (SVMs) are widely used in classification tasks, and this project focuses on building SVM kernels from scratch. The goal is to create decision boundaries that effectively separate classes in high-dimensional data.

## Methodology

The methodology section delves into the mathematical details of SVMs, discussing the optimization problem, Lagrange dual optimization, and the implementation of various kernel functions. Binary SVM and multiclass SVM approaches are outlined, providing a comprehensive understanding of the project's core concepts.

### Kernel Functions

Four different kernel functions are implemented:

- Linear Kernel
- Sigmoid Kernel
- Polynomial Kernel
- Radial Basis Function (RBF) Kernel

These kernels play a crucial role in transforming the input data to higher-dimensional spaces, allowing for more complex decision boundaries.

### Binary SVM

The binary SVM approach is detailed, emphasizing the steps involved in finding optimal parameters for the hyperplane decision boundary. The Lagrange dual optimization problem and the use of kernel tricks are explained.

### Multiclass SVM

The extension to multiclass classification is explored, using a one-vs-others approach. The meta-algorithm for multiclass SVMs is outlined, and the complexity of decision boundaries is discussed.

## Data Overview

The project utilizes an E. coli dataset with seven predictors and class labels. The dataset is imbalanced, and a test/train split is employed for evaluation. Principal Component Analysis (PCA) is used for dimensionality reduction, aiding in the visualization of decision boundaries.

## Results and Discussion

This section presents the training times and accuracy of SVM kernels. Both scikit-learn and custom implementations are compared, revealing that the custom implementation often outperforms scikit-learn but with increased training time. Decision boundaries are visualized in 2D using PCA, showcasing the impact of different kernels on classification. See the paper.pdf for specific results and discussion.

## Conclusion

The project concludes with a summary of key findings, emphasizing the effectiveness of custom SVM kernels and the trade-off between accuracy and training time. Decision boundaries visually illustrate the diverse shapes achievable with various kernels.

Feel free to explore the detailed documentation in the Jupyter notebook for a deeper understanding of the project's implementation and results.