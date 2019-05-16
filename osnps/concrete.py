import numpy as np
from random import random as rand
from math import floor
from numba import cuda

from osnps.base import OSNPS


class SerialOSNPS(OSNPS):
    def __init__(self, *args, **kwargs):
        super(SerialOSNPS, self).__init__(*args, **kwargs)

    def generate_learning_params(self):
        self.a = 0.15 * np.random.random_sample(self.m) + 0.05
        self.delta = 0.015 * rand() + 0.005

    def spike(self, P, T):
        for i in range(self.H):
            for j in range(self.m):
                if (rand() < P[i,j]):
                    T[i,j] = 1
                else:
                    T[i,j] = 0

    def guide(self, P, T, F, F_argmax):
        for i in range(self.H):
            for j in range(self.m):
                if (rand() < self.a[j]):
                    k1, k2 = i, i
                    while (k1 == i or k2 == i):
                        k1, k2 = [floor(rand() * self.H) for i in range(2)]
                    b = T[k1,j] if F[k1] > F[k2] else T[k2,j]
                    P[i,j] = P[i,j] + self.delta if b > 0.5 else P[i,j] - self.delta
                else:
                    P[i,j] = P[i,j] + self.delta if T[F_argmax,j] > 0.5 else P[i,j] - self.delta
                
                # Adjustments
                if P[i,j] > 1:
                    P[i,j] -= self.delta
                if P[i,j] < 0:
                    P[i,j] += self.delta

    def run(self, runs):
        self.P = np.random.random_sample((self.H, self.m))
        self.generate_learning_params()

        self.T = np.zeros((self.H, self.m))
        self.spike(self.P, self.T)
        F = self.fitness_vector(self.T)

        ave_fitness = []
        ave_fitness.append(np.mean(F))

        max_fitness = []
        max_fitness.append(np.max(F))
        
        for r in range(runs):
            self.guide(self.P, self.T, F, self.fitness_argmax(F))
            self.spike(self.P, self.T)
            F = self.fitness_vector(self.T)
            ave_fitness.append(np.mean(F))
            max_fitness.append(np.max(F))
            
        return ave_fitness, max_fitness
