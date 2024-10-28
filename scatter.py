#!/usr/bin/env python3

import mglearn
import matplotlib.pyplot as plt

def plot(X, y):
    plt.scatter(X[:,0], X[:,1], c=y, s=60, cmap=mglearn.cm2)
    plt.show()

# main block
from forge import forge_datasets

if __name__ == '__main__':
    X, y = forge_datasets()
    print("X.shape: %s" % (X.shape,))
    plot(X, y)
