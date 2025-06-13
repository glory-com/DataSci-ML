import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

df = pd.read_csv("data_linear.csv")

def draw_line(k,b) :
    x = np.linspace(-5,5,1000)
    y = k * x + b 
    return x , y 

class LinearRegression :
    def __init__(self , epoches , learning_rate):
        self.epoches = epoches
        self.learning_rate = learning_rate

        self.k = np.random.randint(10)
        self.b = np.random.randint(10)

    def train(self , df):
        for epoch in range(self.epoches) :
            X = df['X']
            y = df['y']
            y_pred = X * self.k + self.b 
            dk = (2 / len(X)) * np.dot(X.T, (y_pred - y))
            db = (2 / len(X)) * np.sum(y_pred - y)

            self.k -= self.learning_rate * dk 
            self.b -= self.learning_rate * db 
        
            if epoch % 50 == 0 :
                plt.scatter(X,y)
                plt.xlabel("X")
                plt.ylabel('y')
                plt.plot(draw_line(self.k , self.b)[0] , draw_line(self.k , self.b)[1] , c='red')
                plt.show(block = False)
                plt.pause(0.5)
                plt.close()

lr = LinearRegression(1000,0.01)
lr.train(df)     


        
