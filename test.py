import matplotlib.pylab as plt 
import numpy as np 
import pandas as pd
import random 


data = pd.read_csv("data.csv")
print(data['X'].iloc[1])