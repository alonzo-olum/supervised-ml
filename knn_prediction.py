#!/usr/bin/env python3

from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, knn):
        self.knn = knn

    @classmethod
    def build_model(cls, X_train, y_train):
        knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                               weights='uniform')
        knn.fit(X_train, y_train)
        return cls(
                knn = knn
                )
        
    def predict(self, X_new):
        return self.knn.predict(X_new)

    def evaluate(self, X_test, y_test):
        return self.knn.score(X_test, y_test)

# predict new data using our knn model

from iris_lib import iris, train_test_data
data = iris()
X_train, _, y_train, _ = train_test_data(data, 'data', 'target')
knn = KNN.build_model(X_train, y_train)

import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("\nPredicted species: %s" % data['target_names'][prediction])
