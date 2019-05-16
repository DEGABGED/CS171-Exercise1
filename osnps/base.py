import numpy as np
from random import random as rand


class KnapsackProblem(object):
    def __init__(self, m):
        self.omega = m*3
        self.K = m
        self.w = np.random.randint(1, self.omega, self.K)
        self.p = self.w + 0.5*self.omega
        self.C = 0.5*np.sum(self.w)

    def fitness(self):
        def f_knapsack(arr):
            knapsack = np.sum(np.multiply(arr, self.p))
            load = np.sum(np.multiply(arr, self.w))
            return knapsack if load <= self.C else -1
        
        return f_knapsack


class OSNPS(object):
    def __init__(self, H, m, fitness=np.sum):
        self.H = H  # Number of ESNPS
        self.m = m  # Number of membranes / bits
        self.fitness = fitness  # Fitness function

    def fitness_vector(self, T):
        return [self.fitness(T[i]) for i in range(self.H)]

    def fitness_argmax(self, F):
        arg = 0
        for i in range(1, self.H):
            arg = i if F[i] > F[arg] else arg
        return arg