import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import cross_val_score, cross_validate, KFold


class Pen_Based_Recognition(object):
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

        self.test_X, self.test_y = self.get_X_y(self.test_path)
        self.train_X, self.train_y = self.get_X_y(self.train_path)

        self.X_data, self.y_data = np.concatenate([self.train_X, self.test_X]), np.concatenate([self.train_y, self.test_y])

        print(self.X_data.shape)
        print(self.y_data.shape)

    def get_X_y(self, path):
        X = []
        y = []
        with open(path) as f_train:
            for line in f_train:
                data_row = line.split(',')
                X.append(data_row[:-1])
                y.append(data_row[-1])
        X = np.asarray(X, dtype=np.float)
        y = np.asarray(y, dtype=np.float)
        return X, y


if __name__ == '__main__':

    pen_based_recognition = Pen_Based_Recognition(train_path='raw_data/pendigits.tra',
                                                  test_path='raw_data/pendigits.tes')


    # iris = datasets.load_iris()
    # iris_X = iris.data
    # iris_y = iris.target
    # print(iris_X.shape)
    # print(iris_y.shape)
    # print('Number of classes: %d' % len(np.unique(iris_y)))
    # print('Number of data points: %d' % len(iris_y))
    #
    # X0 = iris_X[iris_y == 0, :]
    # print('\nSamples from class 0:\n', X0[:5, :])
    #
    # X1 = iris_X[iris_y == 1, :]
    # print('\nSamples from class 1:\n', X1[:5, :])
    #
    # X2 = iris_X[iris_y == 2, :]
    # print('\nSamples from class 2:\n', X2[:5, :])
    #
    clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
    # scores = cross_validate(clf,
    #                         pen_based_recognition.train_data,
    #                         pen_based_recognition.test_data,
    #                         cv=10,
    #                         return_estimator=True)
    # print(scores['test_score'])
    # print(scores.keys())
    # model = scores['estimator']
    # print(model)
    # print(model.score(pen_based_recognition.train_data, pen_based_recognition.test_data))

    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(pen_based_recognition.X_data, pen_based_recognition.y_data)

    for train_index, test_index in kf.split(pen_based_recognition.X_data):
        print("TRAIN:", train_index, "TEST:", test_index)
        # print(len(train_index))
        # print(len(test_index))
        X_train, X_test = pen_based_recognition.X_data[train_index], pen_based_recognition.X_data[test_index]
        y_train, y_test = pen_based_recognition.y_data[train_index], pen_based_recognition.y_data[test_index]

        clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))