import numpy as np
from random import random as rand
from math import floor
from numba import cuda

from osnps.base import OSNPS


class SerialOSNPS(OSNPS):
    def __init__(self, *args, **kwargs):
        super(SerialOSNPS, self).__init__(*args, **kwargs)

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


class ParallelOSNPS(OSNPS):
    def __init__(self, *args, **kwargs):
        super(ParallelOSNPS, self).__init__(*args, **kwargs)

    def set_GPU_params(self, threads_per_block, blocks_per_grid):
        self.threads_per_block = threads_per_block
        self.blocks_per_grid = blocks_per_grid

    @cuda.jit
    def spike(P, T, H, m, Rand):
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        # Set as squares
        block_size_x = cuda.blockDim.x
        block_size_y = cuda.blockDim.y
        grid_size_x = cuda.gridDim.x
        grid_size_y = cuda.gridDim.y

        # For matrices larger than the grid
        x = block_size_x * bx + tx
        y = block_size_y * by + ty

        if (x < P.shape[0] and y < P.shape[1]):
            T[x,y] = 1 if Rand[0,x,y] < P[x,y] else 0

    @cuda.jit
    def guide(P, T, F, F_argmax, H, m, a, delta, Rand):
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        # Set as squares
        block_size_x = cuda.blockDim.x
        block_size_y = cuda.blockDim.y
        grid_size_x = cuda.gridDim.x
        grid_size_y = cuda.gridDim.y

        # For matrices larger than the grid
        x = block_size_x * bx + tx
        y = block_size_y * by + ty
        
        # Redefine some things
        H, m = P.shape

        if (x < H and y < m):
            if(Rand[1,x,y] < a[y]):
                k1 = (int)(Rand[2,x,y]*H)
                k2 = (int)(Rand[3,x,y]*H)
                if F[k1] > F[k2]:
                    b = T[k1,y]
                else:
                    b = T[k2,y]
                P[x,y] = P[x,y] + delta if b > 0.5 else P[x,y] - delta
            else:
                P[x,y] = P[x,y] + delta if T[F_argmax,y] > 0.5 else P[x,y] - delta

            if P[x,y] > 1:
                P[x,y] -= delta
            if P[x,y] < 0:
                P[x,y] += delta

    def run(self, runs):

        # Define the initial probability array and learning rate
        self.P = np.random.random_sample((self.H, self.m))
        P_device = cuda.to_device(self.P)

        self.generate_learning_params()
        a_device = cuda.to_device(self.a)
        
        self.T = np.zeros((self.H, self.m))
        T_device = cuda.to_device(self.T)
        Rand = np.random.random_sample((1, self.H, self.m))
        Rand_device = cuda.to_device(Rand)

        self.spike[self.blocks_per_grid, self.threads_per_block](P_device, T_device, self.H, self.m, Rand_device)
        F = self.fitness_vector(T_device.copy_to_host())

        ave_fitness = []
        ave_fitness.append(np.mean(F))

        max_fitness = []
        max_fitness.append(np.max(F))
        
        for r in range(runs):
            Rand = np.random.random_sample((4, self.H, self.m))
            Rand_device = cuda.to_device(Rand)

            F_argmax = self.fitness_argmax(F)
            F_device = cuda.to_device(F)

            self.guide[self.blocks_per_grid, self.threads_per_block](P_device, T_device, F_device, F_argmax, self.H, self.m, a_device, self.delta, Rand_device)

            self.spike[self.blocks_per_grid, self.threads_per_block](P_device, T_device, self.H, self.m, Rand_device)

            F = self.fitness_vector(T_device.copy_to_host())
            ave_fitness.append(np.mean(F))
            max_fitness.append(np.max(F))
            
        return ave_fitness, max_fitness