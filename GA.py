import matplotlib.pylab as plt 
import numpy as np 
import pandas as pd
import random 

m = [8 , 4 , 2 , 1]

def f(x):
    x1 , x2 , x3 , x4 = x 
    term1_base = x1 - 2
    term2_base = x2 + 3
    term3_base = x3 - 1
    term4_base = x4 + 4
    part_quad_pow = term1_base**4 + term2_base**4 + term3_base**4 + term4_base**4
    part_cube_pow = term1_base**3 - term2_base**3 + term3_base**3 - term4_base**3 # 注意正负号
    part_cross = (x1 - x2)**2 + (x2 - x3)**2 + (x3 - x4)**2
    exp_inner_sum = term1_base**2 + term2_base**2 + term3_base**2 + term4_base**2
    part_exp = np.exp(exp_inner_sum)
    constant = 5
    result = part_quad_pow + part_cube_pow + part_cross + part_exp + constant
    return result


class GA :
    def __init__(self):
        self.func = f 
        self.populationsize = 100 
        self.bound = [-10,10]
        self.population = [np.random.randint(0,2,size=16) for _ in range(self.populationsize)]
        self.mutation_rate = 0.2 
        self.max_it = 1000 

    def mutation(self , dna) :
        pos = np.random.randint(0,16) 
        dna[pos] = 1 if dna[pos] == 0 else 0 

    def code(self , dna) :
        x1 = self.bound[0] + np.dot(dna[0:4] , m) / 15 * (self.bound[1] - self.bound[0])
        x2 = self.bound[0] + np.dot(dna[4:8] , m) / 15 * (self.bound[1] - self.bound[0])
        x3 = self.bound[0] + np.dot(dna[8:12] , m) / 15 * (self.bound[1] - self.bound[0])
        x4 = self.bound[0] + np.dot(dna[12:16] , m) / 15 * (self.bound[1] - self.bound[0])
        return [x1 , x2 , x3 , x4]

    def next_generation(self) :
        self.population = sorted(self.population , key = lambda x : self.func(self.code(x)))
        parents = self.population[:self.populationsize // 5]
        self.population = []

        for i in range(self.populationsize) :
            ind1 = np.random.randint(0 , 20)
            ind2 = np.random.randint(0 , 20)
            father , mother = parents[ind1] , parents[ind2]

            pos = np.random.randint(0,16)
            child = np.concatenate((father[:pos], mother[pos:]))

            if np.random.uniform(0,1) < self.mutation_rate :
                self.mutation(child)

            self.population.append(child)

    def predict(self) :
        for i in range(self.max_it) :
            self.next_generation()
    
    def showInfo(self) :
        print(f"best solution : {self.code(self.population[0])} ")
        print(f"min value : {self.func(self.code(self.population[0]))}")

            
ga = GA()
ga.predict()
ga.showInfo()