import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

data = pd.read_csv("data_svm.csv")

class SVM:
    def __init__(self, data):
        self.data = data
        self.learning_rate = 0.01
        self.epoches = 1000
        self.w = np.random.uniform(-1, 1, size=(2,))  
        self.b = random.uniform(-1, 1)

    def train(self):
        x = self.data[['x', 'y']].values  
        labels = self.data['label'].values
        labels = np.where(labels == 0, -1, 1)
        
        for epoch in range(self.epoches):
            y_pred = np.dot(x, self.w) + self.b
            condition = 1 - labels * y_pred

            mask = (condition > 0).astype(float)

            dw = np.zeros_like(self.w)
            for i in range(len(x)):
                if mask[i] > 0:
                    dw += -labels[i] * x[i]
            dw /= len(x)

            db = np.sum(-labels * mask) / len(x)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {np.mean(np.maximum(0, condition)):.4f}")
                self.draw()

    def draw(self):
        x0 = self.data[self.data['label'] == 0]['x']
        y0 = self.data[self.data['label'] == 0]['y']
        x1 = self.data[self.data['label'] == 1]['x']
        y1 = self.data[self.data['label'] == 1]['y']

        plt.scatter(x0, y0, c='red')
        plt.scatter(x1, y1, c='blue')

        x_plot = np.linspace(min(self.data['x']), max(self.data['x']), 1000)
        y_plot = (-self.b - self.w[0] * x_plot) / self.w[1]
        plt.plot(x_plot, y_plot, c='black')
        plt.xlim(0, 6)
        plt.ylim(0, 6)  
        plt.show()

svm = SVM(data)
svm.train()
