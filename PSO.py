import numpy as np 

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



class PSO :
    def __init__(self):
        self.func = f 
        self.x = np.random.randn(4)
        self.wmax = 1.3
        self.wmin = 0.1 
        self.c1 = 2 
        self.c2 = 2
        self.bound = [-20,20]
        self.vmax = 10 
        self.max_it = 50000
        self.population = 10000
        self.delta = 1 / self.max_it 

        self.v = [np.random.uniform(-10 , 10 , 4) for _ in  range(self.population)]
        self.pos = [np.random.randn(4) for _ in range(self.population)]

        self.gbest_pos = self.pos[0].copy()
        self.gbest_sco = self.func(self.gbest_pos)

        self.pbest_pos = [p.copy() for p in self.pos]
        self.pbest_sco = [self.func(p) for p in self.pos]

    def update(self) :
        for person in range(self.population) :
            v = self.v[person]
            pos = self.pos[person]
            score = self.func(pos)

            if score < self.gbest_sco :
                self.gbest_pos = pos.copy()
                self.gbest_sco = score

            if score < self.pbest_sco[person] :
                self.pbest_pos[person] = pos.copy()
                self.pbest_sco[person] = score

            r1 = np.random.rand(4)
            r2 = np.random.rand(4)

            self.v[person] = self.w * v + self.c1 * r1 * (np.array(self.pbest_pos[person]) - pos) + self.c2 * r2 * (np.array(self.gbest_pos) - pos)
            self.v[person] = np.clip(self.v[person] , -self.vmax , self.vmax)
            self.pos[person] = self.pos[person] + self.v[person]
            self.pos[person] = np.clip(self.pos[person], self.bound[0], self.bound[1])
    
    def predict(self) :
        for it in range(self.max_it) :      
            self.w = self.wmax - it * (self.wmax - self.wmin) / self.max_it 
            self.update()
            self.c1 -= self.delta 
            self.c2 += self.delta 
        return self.gbest_pos, self.gbest_sco
    
pso = PSO()
best_pos , best_score = pso.predict()
print(f"best_pos : {best_pos} , best_score : {best_score}")

