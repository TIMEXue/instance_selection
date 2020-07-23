import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import cross_val_score, cross_validate, KFold

from abc import ABCMeta, abstractmethod


class DatasetBase(object, metaclass=ABCMeta):
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

    pen_based_recognition = DatasetBase(train_path='raw_data/pendigits.tra',
                                        test_path='raw_data/pendigits.tes')

    clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)

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
