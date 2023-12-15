import os
import time # time calculations
import numpy as np # matrix operations
import pandas as pd # loading data
from sklearn.model_selection import train_test_split # data splitting
from sklearn.metrics import confusion_matrix # evaluation metric
import matplotlib.pyplot as plt # for visualization
import seaborn as sns # for visualization
from colors import Color as c # for text colors
from sklearn.decomposition import PCA # for data reduction
# c.examples()


def load_data(log=True, test_size=0.2, random_state=42, binaryClasses=['cp', 'im'], reduce=False):
    print(c.colorize(c.BRIGHT_GREEN, 'Loading Data...'))
    print("test_size: ", test_size)
    print("random_state: ", random_state)
    print("\n")
    # Load Data
    column_names = ['seq_name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
    df = pd.read_csv('data/ecoli.data', sep='\s+', names=column_names)

    # Reduce Data
    if reduce:
        pca = PCA(n_components=2)
        df_reduced = pca.fit_transform(df.drop(['seq_name', 'class'], axis=1))
        df_reduced = pd.DataFrame(df_reduced)
        df_reduced['class'] = df['class']
        df = df_reduced
    else:
        df = df.drop(['seq_name'], axis=1)

    # Binary Classification Dataset
    df_binary = df[df['class'].isin(binaryClasses)] # Selecting only two classes
    X_binary = df_binary.drop(['class'], axis=1)
    X_binary = X_binary.astype(float)
    X_binary = X_binary.values
    y_binary = df_binary['class']
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(X_binary, y_binary, test_size=test_size, random_state=random_state)

    # Multi-class Classification Dataset
    X_multi = df.drop(['class'], axis=1)
    X_multi = X_multi.astype(float)
    X_multi = X_multi.values
    y_multi = df['class']
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=test_size, random_state=random_state)

    # Print Data Information
    if log:
        print(c.colorize(c.BRIGHT_CYAN, 'Class Frequency:'))
        print(df['class'].value_counts())
        print(c.colorize(c.BRIGHT_CYAN, '\nMulti-class Dataset:'))
        print('X_train_multi shape: ', X_train_multi.shape)
        print('y_train_multi shape: ', y_train_multi.shape)
        print('X_test_multi shape: ', X_test_multi.shape)
        print('y_test_multi shape: ', y_test_multi.shape)
        print(c.colorize(c.BRIGHT_CYAN, '\nBinary-class Dataset:'))
        print('X_train_binary shape: ', X_train_binary.shape)
        print('y_train_binary shape: ', y_train_binary.shape)
        print('X_test_binary shape: ', X_test_binary.shape)
        print('y_test_binary shape: ', y_test_binary.shape)
    return X_train_binary, X_test_binary, y_train_binary, y_test_binary, X_train_multi, X_test_multi, y_train_multi, y_test_multi

def calcAccuracy(y_test, y_pred, decimal_places=4):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    return round(accuracy, decimal_places)

def runTests(X_train_binary, X_test_binary, y_train_binary, y_test_binary, X_train_multi, X_test_multi, y_train_multi, y_test_multi, max_iter=1000, C=1.0, tol=0.001, gamma=1, r=1, degree=3, runs=5):
    print(c.colorize(c.BRIGHT_GREEN, 'Running Tests...'))
    from sklearn.svm import SVC
    binary_classes = np.unique(np.concatenate((y_train_binary, y_test_binary)))
    parameters = {
        'runs': [runs],
        'C': [C],
        'max_iter': [max_iter],
        'tol': [tol],
        'gamma': [gamma],
        'r': [r],
        'degree': [degree],
        'binary_classes': [binary_classes]
    }
    parameters = pd.DataFrame(parameters)
    # print(parameters)
    table = pd.DataFrame(columns=['Classifier', 'Implementation', 'Kernel', 'Average Accuracy', 'Average Runtime'])
    # Binary Classification
    # Linear Kernel
    # Sklearn
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = SVC(C=C, kernel='linear', max_iter=max_iter, tol=tol)
        start = time.time()
        classifier.fit(X_train_binary, y_train_binary)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_binary)
        accuracy.append(calcAccuracy(y_test_binary, y_pred))
    table.loc[len(table)] = ['Binary', 'sklearn', 'linear', np.mean(accuracy), np.mean(runtime)]
    # Custom
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = BinarySVM(C=C, kernel_type='linear', max_iter=max_iter, tol=tol)
        start = time.time()
        classifier.fit(X_train_binary, y_train_binary)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_binary)
        accuracy.append(calcAccuracy(y_test_binary, y_pred))
    table.loc[len(table)] = ['Binary', 'custom', 'linear', np.mean(accuracy), np.mean(runtime)]
    # Sigmoid Kernel
    # Sklearn
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = SVC(C=C, kernel='sigmoid', max_iter=max_iter, tol=tol, gamma=gamma, coef0=r)
        start = time.time()
        classifier.fit(X_train_binary, y_train_binary)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_binary)
        accuracy.append(calcAccuracy(y_test_binary, y_pred))
    table.loc[len(table)] = ['Binary', 'sklearn', 'sigmoid', np.mean(accuracy), np.mean(runtime)]
    # Custom
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = BinarySVM(C=C, kernel_type='sigmoid', max_iter=max_iter, tol=tol, gamma=gamma, r=r)
        start = time.time()
        classifier.fit(X_train_binary, y_train_binary)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_binary)
        accuracy.append(calcAccuracy(y_test_binary, y_pred))
    table.loc[len(table)] = ['Binary', 'custom', 'sigmoid', np.mean(accuracy), np.mean(runtime)]
    # RBF Kernel
    # Sklearn
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = SVC(C=C, kernel='rbf', max_iter=max_iter, tol=tol, gamma=gamma)
        start = time.time()
        classifier.fit(X_train_binary, y_train_binary)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_binary)
        accuracy.append(calcAccuracy(y_test_binary, y_pred))
    table.loc[len(table)] = ['Binary', 'sklearn', 'rbf', np.mean(accuracy), np.mean(runtime)]
    # Custom
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = BinarySVM(C=C, kernel_type='rbf', max_iter=max_iter, tol=tol, gamma=gamma)
        start = time.time()
        classifier.fit(X_train_binary, y_train_binary)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_binary)
        accuracy.append(calcAccuracy(y_test_binary, y_pred))
    table.loc[len(table)] = ['Binary', 'custom', 'rbf', np.mean(accuracy), np.mean(runtime)]
    # Polynomial Kernel
    # Sklearn
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = SVC(C=C, kernel='poly', max_iter=max_iter, tol=tol, gamma=gamma, degree=degree, coef0=r)
        start = time.time()
        classifier.fit(X_train_binary, y_train_binary)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_binary)
        accuracy.append(calcAccuracy(y_test_binary, y_pred))
    table.loc[len(table)] = ['Binary', 'sklearn', 'poly', np.mean(accuracy), np.mean(runtime)]
    # Custom
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = BinarySVM(C=C, kernel_type='poly', max_iter=max_iter, tol=tol, gamma=gamma, r=r, degree=degree)
        start = time.time()
        classifier.fit(X_train_binary, y_train_binary)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_binary)
        accuracy.append(calcAccuracy(y_test_binary, y_pred))
    table.loc[len(table)] = ['Binary', 'custom', 'poly', np.mean(accuracy), np.mean(runtime)]
    # Multi-Class Classification
    # Linear Kernel
    # Sklearn
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = SVC(C=C, kernel='linear', max_iter=max_iter, tol=tol)
        start = time.time()
        classifier.fit(X_train_multi, y_train_multi)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_multi)
        accuracy.append(calcAccuracy(y_test_multi, y_pred))
    table.loc[len(table)] = ['Multi', 'sklearn', 'linear', np.mean(accuracy), np.mean(runtime)]

    # Custom
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = MultiSVM(C=C, kernel_type='linear', max_iter=max_iter, tol=tol)
        start = time.time()
        classifier.fit(X_train_multi, y_train_multi)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_multi)
        accuracy.append(calcAccuracy(y_test_multi, y_pred))
    table.loc[len(table)] = ['Multi', 'custom', 'linear', np.mean(accuracy), np.mean(runtime)]

    # Sigmoid Kernel
    # Sklearn
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = SVC(C=C, kernel='sigmoid', max_iter=max_iter, tol=tol, gamma=gamma, coef0=r)
        start = time.time()
        classifier.fit(X_train_multi, y_train_multi)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_multi)
        accuracy.append(calcAccuracy(y_test_multi, y_pred))
    table.loc[len(table)] = ['Multi', 'sklearn', 'sigmoid', np.mean(accuracy), np.mean(runtime)]

    # Custom
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = MultiSVM(C=C, kernel_type='sigmoid', max_iter=max_iter, tol=tol, gamma=gamma, r=r)
        start = time.time()
        classifier.fit(X_train_multi, y_train_multi)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_multi)
        accuracy.append(calcAccuracy(y_test_multi, y_pred))
    table.loc[len(table)] = ['Multi', 'custom', 'sigmoid', np.mean(accuracy), np.mean(runtime)]

    # RBF Kernel
    # Sklearn
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = SVC(C=C, kernel='rbf', max_iter=max_iter, tol=tol, gamma=gamma)
        start = time.time()
        classifier.fit(X_train_multi, y_train_multi)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_multi)
        accuracy.append(calcAccuracy(y_test_multi, y_pred))
    table.loc[len(table)] = ['Multi', 'sklearn', 'rbf', np.mean(accuracy), np.mean(runtime)]

    # Custom
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = MultiSVM(C=C, kernel_type='rbf', max_iter=max_iter, tol=tol, gamma=gamma)
        start = time.time()
        classifier.fit(X_train_multi, y_train_multi)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_multi)
        accuracy.append(calcAccuracy(y_test_multi, y_pred))
    table.loc[len(table)] = ['Multi', 'custom', 'rbf', np.mean(accuracy), np.mean(runtime)]

    # Polynomial Kernel
    # Sklearn
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = SVC(C=C, kernel='poly', max_iter=max_iter, tol=tol, gamma=gamma, degree=degree, coef0=r)
        start = time.time()
        classifier.fit(X_train_multi, y_train_multi)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_multi)
        accuracy.append(calcAccuracy(y_test_multi, y_pred))
    table.loc[len(table)] = ['Multi', 'sklearn', 'poly', np.mean(accuracy), np.mean(runtime)]

    # Custom
    runtime = []
    accuracy = []
    for i in range(runs):
        classifier = MultiSVM(C=C, kernel_type='poly', max_iter=max_iter, tol=tol, gamma=gamma, r=r, degree=degree)
        start = time.time()
        classifier.fit(X_train_multi, y_train_multi)
        end = time.time()
        runtime.append(round(end-start, 4))
        y_pred = classifier.predict(X_test_multi)
        accuracy.append(calcAccuracy(y_test_multi, y_pred))
    table.loc[len(table)] = ['Multi', 'custom', 'poly', np.mean(accuracy), np.mean(runtime)]
    return table, parameters

class BinarySVM:
    def __init__(self, max_iter=1000, kernel_type='linear', C=1.0, tol=0.001, gamma=0.1, r=1, degree=3):
        self.kernels = {
            'linear' : self.kernel_linear,
            'sigmoid': self.kernel_sigmoid,
            'rbf': self.kernel_rbf,
            'poly': self.kernel_polynomial
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.tol = tol
        self.gamma = gamma
        self.r = r
        self.degree = degree
        self.y = None
        self.y_encoded = None
        self.classes = None
    def encode_labels(self, y):
        self.classes = np.unique(y)
        self.y = y.copy()
        self.y_encoded = np.where(self.y == self.classes[0], -1, 1)
    def fit(self, X, y):
        # Initialization
        n, d = X.shape[0], X.shape[1]
        self.encode_labels(y)
        alpha = np.zeros((n))
        kernel = self.kernels[self.kernel_type]
        count = 0
        while True:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, n):
                i = self.get_random_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], self.y_encoded[i], self.y_encoded[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_L_H(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, self.y_encoded, X)
                self.b = self.calc_b(X, self.y_encoded, self.w)

                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.tol:
                break

            if count >= self.max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self.max_iter))
                return
        # Compute final model parameters
        self.b = self.calc_b(X, self.y_encoded, self.w)
        if self.kernel_type == 'linear':
            self.w = self.calc_w(alpha, self.y_encoded, X)
        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count

    def predict(self, X):
        predictions = self.h(X, self.w, self.b)
        return np.where(predictions == -1, self.classes[0], self.classes[1])

    def calc_b(self, X, y, w):
        b_tmp = y - np.dot(w.T, X.T)
        return np.mean(b_tmp)

    def calc_w(self, alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))

    # Prediction
    def h(self, X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    def compute_L_H(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if(y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def get_random_int(self, a, b, z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = np.random.randint(a, b)
            cnt=cnt+1
        return i

    # Define kernels
    def kernel_linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def kernel_sigmoid(self, x1, x2):
        return np.tanh(self.gamma * np.dot(x1, x2.T) + self.r)

    def kernel_rbf(self, x1, x2):
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-self.gamma * distance)
    def kernel_polynomial(self, x1, x2):
        return (np.dot(x1, x2) + self.r) ** self.degree

class MultiSVM:
    def __init__(self, C=1.0, max_iter=1000, kernel_type='linear', tol=0.001, gamma=0.1, r=1, degree=3):
        self.C = C
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.tol = tol
        self.gamma = gamma
        self.r = r
        self.models = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            # Create a binary label for this class
            binary_y = (y == c).astype(int)
            binary_y[binary_y == 0] = -1
            # Train a binary SVM
            svm = BinarySVM(C=self.C, max_iter=self.max_iter, kernel_type=self.kernel_type, tol=self.tol, gamma=self.gamma, r=self.r)
            svm.fit(X, binary_y)
            # Save the trained model
            self.models.append(svm)

    def predict(self, X):
        # Get the prediction from each binary SVM
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, svm in enumerate(self.models):
            predictions[:, i] = svm.predict(X)
        # Choose the class that got the most votes
        return self.classes[np.argmax(predictions, axis=1)]

# X_train_binary, X_test_binary, y_train_binary, y_test_binary, X_train_multi, X_test_multi, y_train_multi, y_test_multi = load_data(log=True, binaryClasses=['cp', 'om'])
# runTests(X_train_binary, X_test_binary, y_train_binary, y_test_binary, X_train_multi, X_test_multi, y_train_multi, y_test_multi)

