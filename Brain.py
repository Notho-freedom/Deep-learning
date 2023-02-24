import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = X.dot(W) + b
    # print(Z.min())
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    # print(A)
    return A >= 0.5

from tqdm import tqdm

def artificial_neuron(X_train, y_train, X_test, y_test, learning_rate = 0.1, n_iter = 100):
    # initialisation W, b
    W, b = initialisation(X_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in tqdm(range(n_iter)):
        A = model(X_train, W, b)

        if i %10 == 0:
            # Train
            train_loss.append(log_loss(A, y_train))
            y_pred = predict(X_train, W, b)
            train_acc.append(accuracy_score(y_train, y_pred))

            # Test
            A_test = model(X_test, W, b)
            test_loss.append(log_loss(A_test, y_test))
            y_pred = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred))

        # mise a jour
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)


#pip install h5py # h5py vous permet d'ouvrir les fichiers au format hdf5. N'oubliez pas de l'installer !
from utilities import *

X_train, y_train, X_test, y_test = load_data()

print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

print(X_test.shape)
print(y_test.shape)
print(np.unique(y_test, return_counts=True))



X_train_reshape = X_train.reshape(X_train.shape[0], -1) / X_train.max()
print(X_train_reshape.shape)

X_test_reshape = X_test.reshape(X_test.shape[0], -1) / X_train.max()
print(X_test_reshape.shape)

#W, b = artificial_neuron(X_train_reshape, y_train, X_test_reshape, y_test, learning_rate = 0.01, n_iter=10000)
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
X[:, 1] = X[:, 1] * 1

def artificial_neuron_2(X, y, learning_rate=0.1, n_iter=1000):

    W, b = initialisation(X)
    W[0], W[1] = -7.5, 7.5

    nb = 1
    j=0
    A = model(X, W, b)
    Loss = []
    

    Params1 = [W[0]]
    Params2 = [W[1]]
    Loss.append(log_loss(y, A))
    
    # Training
    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(log_loss(y, A))
        Params1.append(W[0])
        Params2.append(W[1])
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate = learning_rate)

    return b, Loss, Params1, Params2


b, Loss, Params1, Params2 = artificial_neuron_2(X, y, learning_rate=0.6, n_iter=100)










#===========================================================================================================================
