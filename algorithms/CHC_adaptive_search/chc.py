import random
import time
from typing import List
import numpy as np
from scipy.spatial import distance
from sklearn import neighbors

np.random.seed(1122)
random.seed(1122)


counting_fitness = 0

class CHC:
    """
    paper: https://www.sciencedirect.com/science/article/pii/B9780080506845500203
    """
    def __init__(self, n_generation: int,
                       n_population: int,
                       divergence_rate: float,
                       X_data,
                       y_data,
                       alpha):

        self.n_population = n_population
        self.n_generation = n_generation
        self.candidate_offspring = list()
        self.candidate_offspring_apostrophe = list()
        self.population = list()
        self.c_generation = 0
        self.divergence_rate = divergence_rate
        self.X_data = X_data
        self.y_data = y_data
        self.n_gene = self.X_data.shape[0]
        self.d = self.n_gene // 4
        self.alpha = alpha

    def initiation(self):
        "Tested"
        for iteration in range(self.n_population):
            P = np.random.randint(2, size=self.n_gene)
            self.population.append(P)

    def fitness(self, subset):
        start_time = time.time()
        global counting_fitness
        counting_fitness += 1
        clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
        subset = subset.astype('bool')
        subset_of_X = self.X_data[subset]
        subset_of_y = self.y_data[subset]
        perc_red = 100.0 * (self.n_gene - np.count_nonzero(subset)) / self.n_gene
        mask = np.ones(subset_of_X.shape[0], bool)
        scores_v = []
        for i in range(subset_of_X.shape[0]):
            mask[i] = False
            clf.fit(subset_of_X[mask], subset_of_y[mask])
            _score = clf.predict(np.expand_dims(subset_of_X[i], axis=0))
            scores_v.append(int(int(_score) == subset_of_y[i]))
            mask[i] = True

        class_rat = np.mean(scores_v)
        fitness_v = self.alpha * class_rat + (1 - self.alpha) * perc_red
        total = 3 * len(self.candidate_offspring_apostrophe) + len(self.population)
        print("Number: {:d}/{:d}, class_rat: {:.2f}, perc_red: {:.2f}, fitness: {:.2f}, running time: {:.2f}".format(counting_fitness, total,
                                                                                               class_rat, perc_red, fitness_v, time.time() - start_time))
        return fitness_v

    def select_r(self):
        "Tested"
        print("Select_r progress --------------------------")
        self.c_generation += 1
        self.candidate_offspring = self.population.copy()
        order = np.random.permutation(self.n_population)
        self.candidate_offspring = [self.candidate_offspring[index] for index in order]

    def recombine(self):
        "Tedted"
        print("Recombine progress --------------------------")
        self.candidate_offspring_apostrophe = self.candidate_offspring.copy()
        saved_index = []
        # print(self.n_population)
        for iteration in range(0, self.n_population, 2):
            hamming_distance = self.n_gene * distance.hamming(self.candidate_offspring_apostrophe[iteration],
                                                              self.candidate_offspring_apostrophe[iteration + 1])
            hamming_distance = int(hamming_distance)
            # print("Hamming_distance and 2d: {} {}".format(hamming_distance, 2 * self.d))
            if hamming_distance > 2 * self.d:
                diff = self.candidate_offspring_apostrophe[iteration] != self.candidate_offspring_apostrophe[iteration + 1]
                positions = [idx for idx, e in enumerate(diff) if e]
                # print("len position: {}".format(len(positions)))
                positions = random.sample(positions, hamming_distance // 2)
                # print("len position: {}".format(len(positions)))
                for pos in positions:
                    # print("before")
                    # print(self.candidate_offspring_apostrophe[iteration][pos])
                    # print(self.candidate_offspring_apostrophe[iteration + 1][pos])
                    tmp = self.candidate_offspring_apostrophe[iteration][pos]
                    self.candidate_offspring_apostrophe[iteration][pos] = self.candidate_offspring_apostrophe[iteration + 1][pos]
                    self.candidate_offspring_apostrophe[iteration + 1][pos] = tmp
                    # print("after")
                    # print(self.candidate_offspring_apostrophe[iteration][pos])
                    # print(self.candidate_offspring_apostrophe[iteration + 1][pos])

                saved_index.append(iteration)
                saved_index.append(iteration + 1)

        self.candidate_offspring_apostrophe = [self.candidate_offspring_apostrophe[index] for index in saved_index]

    def select_s(self):
        global counting_fitness
        counting_fitness = 0
        print("Select_s progress --------------------------")
        # print(len(self.candidate_offspring_apostrophe))
        # print(len(self.population))
        fitness_offspring = [-self.fitness(offspring) for offspring in self.candidate_offspring_apostrophe]
        arg_fitness_offspring = np.argsort(fitness_offspring)
        fitness_parents = [-self.fitness(parent) for parent in self.population]
        arg_fitness_parent = np.argsort(fitness_parents)
        c_index = arg_fitness_parent.shape[0] - 1
        have_changed = False
        print("End sorting---------------------------------")
        for index_offspring in arg_fitness_offspring:
            value_offspring = self.fitness(self.candidate_offspring_apostrophe[index_offspring])
            value_parent = self.fitness(self.population[arg_fitness_parent[c_index]])
            if value_parent < value_offspring:
                have_changed = True
                self.population[arg_fitness_parent[c_index]] = self.candidate_offspring_apostrophe[index_offspring]
            else:
                break
            c_index -= 1
        return have_changed

    def diverge(self):
        print("Diverse progress --------------------------")
        save_v = -1
        save_e = None
        for e in self.population:
            fit = self.fitness(e)
            if fit > save_v:
                save_v = fit
                save_e = e

        for _ in range(self.n_population):
            self.population.append(np.copy(save_e))

        bit_split = int(self.divergence_rate * self.n_gene)
        for element in self.population[:-1]:
            order = random.sample([i for i in range(self.n_gene)], bit_split)
            for idx in order:
                element[idx] = 1 - element[idx]

        self.d = self.divergence_rate * (1.0 - self.divergence_rate) * self.n_gene

    def evolve(self):
        self.c_generation = 0
        self.d = self.n_gene // 4
        self.initiation()
        np.save('logs/population{}'.format(0), self.population)
        for generation_idx in range(self.n_generation):
            print("GENERATION: {} ---------------------------------------------------".format(generation_idx))
            self.select_r()
            self.recombine()
            have_changed = self.select_s()
            print("have_changed = {}".format(have_changed))
            if not have_changed:
                self.d -= 1
            if self.d < 0:
                self.diverge()

            np.save('logs/population{}'.format(generation_idx), self.population)