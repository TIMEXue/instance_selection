import random

import numpy as np
from sklearn import neighbors

from datasets.base import DatasetBase
from algorithms.CHC_adaptive_search.chc import CHC
from configs import cfg

if __name__ == '__main__':
    pen_based_recognition = DatasetBase(train_path='datasets/pen_based_recognition/pendigits.tra',
                                        test_path='datasets/pen_based_recognition/pendigits.tes')

    chc = CHC(n_generation=cfg['EVOLUTION']['n_generation'],
              n_population=cfg['EVOLUTION']['n_population'],
              divergence_rate=cfg['EVOLUTION']['divergence_rate'],
              X_data=pen_based_recognition.train_X,
              y_data=pen_based_recognition.train_y,
              alpha=cfg['EVOLUTION']['alpha'])

    populations = np.load('logs/pen_based_recognition/population249.npy')
    # save_v = -1
    # save_population = None
    # save_idx = 0
    # for idx, population in enumerate(populations):
    #     fit = chc.fitness(population)
    #     if fit > save_v:
    #         save_v = fit
    #         save_idx = idx
    #         save_population = population
    #
    # print("Best Population: {}".format(save_idx))
    # for idx in range(save_population.shape[0]):
    #     print("{} ".format(save_population[idx]), end='')
    # print()

    save_population = populations[0]
    print("check")
    print(save_population)


    new_save_population = []
    for idx, e in enumerate(save_population):
        print(e)
        if e == 0:
            save_population[idx] = False
            new_save_population.append(False)
        else:
            save_population[idx] = True
            new_save_population.append(True)
    # print("old")
    save_population = save_population.astype('bool')
    # print(save_population)
    # print("new")
    # print(new_save_population)
    clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
    sub_set_define = []
    for i in range(10):
        sub_set_define.append(True)
    for i in range(10, pen_based_recognition.train_X.shape[0]):
        sub_set_define.append(True)

    random.shuffle(sub_set_define)
    sub_set_define = np.asarray(sub_set_define)

    print(sub_set_define)
    # subset_of_X = pen_based_recognition.train_X[save_population]
    # subset_of_y = pen_based_recognition.train_y[save_population]
    subset_of_X = pen_based_recognition.train_X[new_save_population]
    subset_of_y = pen_based_recognition.train_y[new_save_population]

    print(subset_of_X.shape)
    print(pen_based_recognition.train_X.shape)
    n_gene = pen_based_recognition.train_X.shape[0]
    # perc_red = 100.0 * (n_gene - np.count_nonzero(save_population)) / n_gene

    print(subset_of_X)

    print(pen_based_recognition.train_X)

    clf.fit(subset_of_X, subset_of_y)
    print(clf.score(pen_based_recognition.test_X, pen_based_recognition.test_y))

    clf.fit(pen_based_recognition.train_X, pen_based_recognition.train_y)
    print(clf.score(pen_based_recognition.test_X, pen_based_recognition.test_y))
