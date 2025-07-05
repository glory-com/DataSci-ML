import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 




class DBSCAN :
    def __init__(self , data , r = 1 , mnpt = 3):
        self.data = data 
        self.cores = []
        self.boaders = []
        self.noises = []
        self.r = r 
        self.mnpt = mnpt 

    def classpt(self):
        for i in range(self.data.shape[0]) :
            idx = 0 
            x = self.data.iloc[i]['x']
            y = self.data.iloc[i]['y']
            for j in range(self.data.shape[0]) :
                if j == i :
                    pass 
                x_ = self.data.iloc[j]['x']
                y_ = self.data.iloc[j]['y']              

                if (x_ - x) ** 2 + (y_ - y) ** 2 <= self.r ** 2 :
                    idx += 1 

            if idx >= self.mnpt :
                self.cores.append([x,y])
            elif idx > 0 :
                self.boaders.append([x,y])
            else :
                self.noises.append([x,y])
        
    def draw(self):
        x1 = [x[0] for x in self.cores]
        y1 = [y[1] for y in self.cores]
        x2 = [x[0] for x in self.boaders]
        y2 = [y[1] for y in self.boaders]
        x3 = [x[0] for x in self.noises]
        y3 = [y[1] for y in self.noises]        

        plt.scatter(x1 , y1 , c="red")
        plt.scatter(x2 , y2 , c="blue")
        plt.scatter(x3 , y3 , c="gray")

        plt.show()


df = pd.read_csv("data_density.csv")
d = DBSCAN(df , r = 0.6 , mnpt = 3)
d.classpt()
d.draw()