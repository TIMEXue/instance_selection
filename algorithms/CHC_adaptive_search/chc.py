import random
from typing import List
import numpy as np
from scipy.spatial import distance
from sklearn import neighbors


class CHC:
    """
    paper: https://www.sciencedirect.com/science/article/pii/B9780080506845500203
    """
    def __init__(self, n_generation: int,
                       n_population: int,
                       n_sample: int,
                       size_class: List,
                       divergence_rate: float,
                       X_data,
                       y_data,
                       alpha):

        self.n_population = n_population
        self.n_generation = n_generation
        self.n_gene = sum(size_class)
        self.size_class = size_class
        self.n_sample = n_sample
        self.candidate_offspring = list()
        self.candidate_offspring_apostrophe = list()
        self.population = list()
        self.c_generation = 0
        self.d = self.n_gene // 4
        self.divergence_rate = divergence_rate
        self.X_data = X_data
        self.y_data = y_data
        self.alpha = alpha

    def initiation(self):
        for iteration in range(self.n_population):
            P = np.random.randint(2, size=self.n_gene)
            self.population.append(P)

    def fitness(self, subset):

        clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)

        subset_of_X = self.X_data[subset]
        subset_of_y = self.y_data[subset]

        perc_red = 100.0 * (self.n_gene - np.count_nonzero(subset)) / self.n_gene

        mask = np.ones(subset_of_X.shape, bool)
        scores_v = []
        for i in range(subset_of_X.shape[0]):
            mask[i] = False
            clf.fit(subset_of_X[mask], subset_of_y[mask])
            _score = clf.score(subset_of_X[i], subset_of_y[i])
            scores_v.append(_score)
            mask[i] = True

        class_rat = np.mean(scores_v)
        fitness_v = self.alpha * class_rat + (1 - self.alpha) * perc_red
        return fitness_v

    def select_r(self):
        self.c_generation += 1
        self.candidate_offspring = self.population.copy()
        order = np.random.permutation(self.n_population)
        self.candidate_offspring = [self.candidate_offspring[index] for index in order]

    def recombine(self):
        self.candidate_offspring_apostrophe = self.candidate_offspring.copy()
        saved_index = []
        for iteration in range(self.n_population, 2):
            hamming_distance = self.n_gene * distance.hamming(self.candidate_offspring_apostrophe[iteration],
                                                              self.candidate_offspring_apostrophe[iteration + 1])
            hamming_distance = int(hamming_distance)

            if hamming_distance > 2 * self.d:
                positions = np.where(self.candidate_offspring_apostrophe[iteration] !=
                                     self.candidate_offspring_apostrophe[iteration + 1])
                positions = random.sample(positions, hamming_distance // 2)

                for pos in positions:
                    tmp = self.candidate_offspring_apostrophe[iteration][pos]
                    self.candidate_offspring_apostrophe[iteration][pos] = self.candidate_offspring_apostrophe[iteration + 1][pos]
                    self.candidate_offspring_apostrophe[iteration + 1][pos] = tmp
                saved_index.append(iteration)
                saved_index.append(iteration + 1)

        self.candidate_offspring_apostrophe = [self.candidate_offspring_apostrophe[index] for index in saved_index]

    def select_s(self):
        fitness_offspring = [-self.fitness(offspring) for offspring in self.candidate_offspring_apostrophe]
        arg_fitness_offspring = np.argsort(fitness_offspring)
        fitness_parents = [-self.fitness(parent) for parent in self.population]
        arg_fitness_parent = np.argsort(fitness_parents)
        c_index = arg_fitness_parent.shape[0] - 1
        for index_offspring in arg_fitness_offspring:
            value_offspring = self.candidate_offspring_apostrophe[index_offspring]
            value_parent = self.population[arg_fitness_parent[c_index]]
            if value_parent < value_offspring:
                self.population[arg_fitness_parent[c_index]] = value_offspring
            else:
                break
            c_index -= 1

    def diverge(self):
        save_v = -1
        save_e = None
        for e in self.population:
            if self.fitness(e) > save_v:
                save_v = self.fitness(e)
                save_e = e

        for _ in range(self.n_population):
            self.population.append(np.copy(save_e))

        bit_split = int(self.divergence_rate * self.n_gene)
        for element in self.population[:-1]:
            order = random.sample([i for i in range(self.n_gene)], bit_split)
            for idx in order:
                element[idx] = 1 - element[idx]

        self.d = self.divergence_rate * (1.0 - self.divergence_rate) * self.n_gene
