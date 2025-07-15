# 粒子群算法

启发式算法是当今主要的研究方向，相比于数学上面对具体问题想要获得精确的解，启发式算法希望在一定的计算量下得到近似的最优解。在本节内容中，先介绍三种经典启发式算法的第一种，粒子群算法。粒子群算法是通过对于自然界中鸟群的行为的模拟，给定一群鸟，和一个确定目标，让鸟自由的飞翔，鸟群往往可以通过群体的最优解找到目标，因此，粒子群算法就是模拟这种过程来找到任务的最优解。

## 实验过程

一、 基本流程

1. 给定目标函数 
在本文的介绍中，我们以找到函数的最小值为任务，通过PSO算法来找到函数最优解，本文中以一个四元三次函数为目标函数。

2. 初始化鸟群
将种群的数量设置为n，随机初始化n个四维向量，获得开始的鸟群位置，再随机初始化n个四维向量，获得开始的鸟群速度。

3. 更新
在PSO算法中，最重要的就是速度和位置的更新。

$$
v_{i+1} = w v_i + c_1 r_1 (gp_{best} - pos) + c_2 r_2 (pp_{best} - pos)\\
$$

$$
\text{其中:}
$$

$$
v_{i+1} \text{表示更新后的位置}
$$

$$
v_i \text{更新前的位置}
$$

$$
w \text{更新权重，表示保留当前位置的比例，前期应该大，后期应该小}
$$

$$
c_1 , c_2 \text{确认的权重，表示探索的欲望，欲望越强，越容易跳出局部最优解}
$$

$$
r_1 , r_2 \text{随机参数，提高随机性}
$$

$$
gp_{best} , pp_{best} \text{表示群体和个体探索到的最优解}
$$

$$
pos \text{当前位置}
$$

$$
pos_{i+1} = pos_i + v_{v+1}
$$

位置更新

3.参数设置
在前期中，最优解是不稳定的，所以我们希望鸟群尽可能地多方向探索，因此前期的w应该尽可能的小。但是在后期中，最优解偏向于稳定，因此，我们希望鸟群尽可能往最优解走，找到精确的最优解。所以，我们需要设置动态的w参数。

$$
w_i = w_{max} - \frac{w_{max} - w_{min}}{max_{iterations}} \cdot iteration 
$$

c的参数也可以根据轮数改变。 

## 完整代码
```python
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

```

## 总结
- PSO算法是一个经典的启发式算法，算法实现简单，效果一般不错，但是要考虑到局部最优化。
- 面对局部最优，可以设计多个鸟群，防止陷入局部而出不来。
- PSO算法的参数较少，主要是参数的灵活性和可变性需要注意，适合初学者。