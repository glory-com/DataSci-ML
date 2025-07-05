# 线性回归
线性回归是统计学，数据科学，人工智能中最关键的一个预测模型，在线性回归中，我们认为在某个任务中，自变量x，因变量y，满足线性关系，希望通过一条直线来表示变量之间的关系。
如何描述预测直线的好坏：通过预测直线与已知点之间的距离的方差。

## 实验过程
一、 定义直线

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

其中：
y是响应变量（因变量）

x是特征（自变量）

 $ \beta_0 $ 是截距 

 $ \beta_1 $ 是斜率 

 $ \epsilon $ 是偏移量


二、 定义损失函数

$$
\text{L($\beta_0$ , $\beta_1$)} = \frac{1}{N} \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i)) ^2}
$$

其中: 

 $ y_i $ 是第i个样本的真实值

 $ x_i $ 是第i个样本的输入

三、 损失最小化
 
对 $ \beta_0 $ 求导：

$$
\frac{\partial L}{\partial \beta_0} = \frac{\partial}{\partial \beta_0} \frac{\sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i)) ^2}}{N}
$$

$$
= -\frac{2}{N}\sum_{i=1}^{N}{(y_i-\beta_0 - \beta_1 x_i)}
$$

对 $\beta_1$ 求导：

$$
\frac{\partial L}{\partial \beta_1} = \frac{\partial}{\partial \beta_1} \frac{\sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i)) ^2}}{N}
$$

$$
= -\frac{2}{N}\sum_{i=1}^{N}{x_i(y_i-\beta_0-\beta_1x_i)}
$$

初始化两个参数 $\beta_0 , \beta_1$ ,计算MSE，通过梯度下降的方式，不断更新参数。

四、代码

```python
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

LinearR = LinearRegression(1000,0.01)
LinearR.train(df)     
```