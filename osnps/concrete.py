import time
import numpy as np
from random import random as rand
from math import floor
from numba import cuda, jit

from osnps.base import OSNPS


class SerialOSNPS(OSNPS):
    def __init__(self, *args, **kwargs):
        super(SerialOSNPS, self).__init__(*args, **kwargs)

    def spike_unopt(self, P, T, H, m):
        for i in range(H):
            for j in range(m):
                if (rand() < P[i,j]):
                    T[i,j] = 1
                else:
                    T[i,j] = 0

    def guide_unopt(self, P, T, F, F_argmax, H, m, a, delta):
        for i in range(H):
            for j in range(m):
                if (rand() < a[j]):
                    k1, k2 = i, i
                    while (k1 == i or k2 == i):
                        k1 = floor(rand() * H)
                        k2 = floor(rand() * H)
                    b = T[k1,j] if F[k1] > F[k2] else T[k2,j]
                    P[i,j] = P[i,j] + delta if b > 0.5 else P[i,j] - delta
                else:
                    P[i,j] = P[i,j] + delta if T[F_argmax,j] > 0.5 else P[i,j] - delta
                
                # Adjustments
                if P[i,j] > 1:
                    P[i,j] -= delta
                if P[i,j] < 0:
                    P[i,j] += delta

    @jit
    def spike(self, P, T, H, m):
        for i in range(H):
            for j in range(m):
                if (rand() < P[i,j]):
                    T[i,j] = 1
                else:
                    T[i,j] = 0

    @jit
    def guide(self, P, T, F, F_argmax, H, m, a, delta):
        for i in range(H):
            for j in range(m):
                if (rand() < a[j]):
                    k1, k2 = i, i
                    while (k1 == i or k2 == i):
                        k1 = floor(rand() * H)
                        k2 = floor(rand() * H)
                    b = T[k1,j] if F[k1] > F[k2] else T[k2,j]
                    P[i,j] = P[i,j] + delta if b > 0.5 else P[i,j] - delta
                else:
                    P[i,j] = P[i,j] + delta if T[F_argmax,j] > 0.5 else P[i,j] - delta
                
                # Adjustments
                if P[i,j] > 1:
                    P[i,j] -= delta
                if P[i,j] < 0:
                    P[i,j] += delta

    def run(self, runs):
        start = time.time()

        self.P = np.random.random_sample((self.H, self.m))
        self.generate_learning_params()

        self.T = np.zeros((self.H, self.m), dtype=np.int8)
        self.spike(self.P, self.T, self.H, self.m)
        F = self.fitness_vector(self.T)

        ave_fitness = []
        ave_fitness.append(np.mean(F))

        max_fitness = []
        max_fitness.append(np.max(F))

        for r in range(runs):
            self.guide(self.P, self.T, F, self.fitness_argmax(F), self.H, self.m, self.a, self.delta)
            self.spike(self.P, self.T, self.H, self.m)
            F = self.fitness_vector(self.T)
            ave_fitness.append(np.mean(F))
            max_fitness.append(np.max(F))

        end = time.time()
        runtime = end - start

        return ave_fitness, max_fitness, runtime

    def run_unopt(self, runs):
        start = time.time()

        self.P = np.random.random_sample((self.H, self.m))
        self.generate_learning_params()

        self.T = np.zeros((self.H, self.m), dtype=np.int8)
        self.spike_unopt(self.P, self.T, self.H, self.m)
        F = self.fitness_vector(self.T)

        ave_fitness = []
        ave_fitness.append(np.mean(F))

        max_fitness = []
        max_fitness.append(np.max(F))

        for r in range(runs):
            self.guide_unopt(self.P, self.T, F, self.fitness_argmax(F), self.H, self.m, self.a, self.delta)
            self.spike_unopt(self.P, self.T, self.H, self.m)
            F = self.fitness_vector(self.T)
            ave_fitness.append(np.mean(F))
            max_fitness.append(np.max(F))

        end = time.time()
        runtime = end - start

        return ave_fitness, max_fitness, runtime

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
        start = time.time()

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

        end = time.time()
        runtime = end - start
        return ave_fitness, max_fitness, runtime