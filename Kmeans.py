import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 


df = pd.read_csv("data_class.csv")
COLOURS = ['red' , 'green' , 'blue' , 'gray' , 'orange']
def calc_dis(y,x) :
    return (y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 

class Points :
    def __init__(self , middle):
        self.middle = middle 
        self.points = []
        

    def add_points(self,p) :
        self.points.append(p)
    
class Kmeans :
    def __init__(self , k , df , epoches):
        self.k = k 
        self.X_BOUND = [df['X'].min(),df['X'].max()]
        self.Y_BOUND = [df['Y'].min(),df['Y'].max()]
        self.epoches = epoches 
        self.df = df 
        self.classes = [Points(self.get_middles(self.X_BOUND , self.Y_BOUND)) for _ in range(self.k)]
        self.last_classes = []
 
    def get_middles(self,bound1,bound2) :
        x = np.random.uniform(bound1[0] , bound1[1])
        y = np.random.uniform(bound2[0] , bound2[1])
        return x , y 
    

    def train(self) :
        for epoch in range(self.epoches) :
            for cls in self.classes :
                cls.points = []

            for row in self.df.itertuples() :
                x , y = row.X , row.Y 
                li = []
                for classes in self.classes :
                    mid = classes.middle 
                    distance = calc_dis((x,y) , mid)
                    li.append(distance)
                ind = li.index(min(li))
                self.classes[ind].add_points([x,y])
            
            self.last_mid = [i.middle for i in self.classes]

            for classes in self.classes :
                if len(classes.points) != 0 :
                    classes.middle = (sum(a[0] for a in classes.points) / len(classes.points) , sum(b[1] for b in classes.points) / len(classes.points))

            self.now_mid = [i.middle for i in self.classes]
            

            print(self.last_mid)
            print(self.now_mid)
            if self.last_mid == self.now_mid :
                break 
            
            self.plot_clusters(epoch)

            
    def plot_clusters(self, epoch):
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(self.classes):
            xs = [p[0] for p in cls.points]
            ys = [p[1] for p in cls.points]
            plt.scatter(xs, ys, c=COLOURS[i % len(COLOURS)], label=f'Cluster {i+1}')

        centers_x = [cls.middle[0] for cls in self.classes]
        centers_y = [cls.middle[1] for cls in self.classes]
        plt.scatter(centers_x, centers_y, s=200, c='black', marker='X', label='Centers')

        plt.title(f"Epoch {epoch}")
        plt.xlabel("X")
        plt.ylabel("Y")

   
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.grid(True)
        plt.show(block=False)
        plt.pause(2) 
        plt.close()



km = Kmeans(2 , df , 1000)
km.train()




                
            