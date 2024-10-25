#!/usr/bin/env python3

import matplotlib.pyplot as plt

def pair_plot(X_train, y_train, data, label):

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle('iris_pairplot')

    for i in range(3):
        for j in range(3):
            ax[i,j].scatter(X_train[:,j], X_train[:, i + 1], c=y_train, s=60)
            ax[i, j].set_xticks(())
            ax[i, j].set_yticks(())
            if i == 2:
                ax[i,j].set_xlabel(data[label][j])
            if j == 0:
                ax[i, j].set_ylabel(data[label][i+1])
            if j > i:
                ax[i, j].set_visible(False)
    plt.show()

# main block

from iris_lib import iris, train_test_data

data = iris()
X_train, _, y_train, _ = train_test_data(data, 'data', 'target')

pair_plot(X_train, y_train, data, 'feature_names')
