# 线性回归

线性回归是统计学，数据科学，人工智能中最关键的一个预测模型，在线性回归中，我们认为在某个任务中，自变量x，因变量y，满足线性关系，希望通过一条直线来表示变量之间的关系。
如何描述预测直线的好坏：通过预测直线与已知点之间的距离的方差。

## 实验过程

一、 定义直线

<img src="https://www.zhihu.com/equation?tex=y%20%3D%20%5Cbeta_0%20%2B%20%5Cbeta_1%20x%20%2B%20%5Cepsilon%5C%5C" alt="y = \beta_0 + \beta_1 x + \epsilon\\" class="ee_img tr_noresize" eeimg="1">

其中：
y是响应变量（因变量）

x是特征（自变量）

<img src="https://www.zhihu.com/equation?tex=%20%5Cbeta_0%20" alt=" \beta_0 " class="ee_img tr_noresize" eeimg="1"> 是截距

<img src="https://www.zhihu.com/equation?tex=%20%5Cbeta_1%20" alt=" \beta_1 " class="ee_img tr_noresize" eeimg="1"> 是斜率

<img src="https://www.zhihu.com/equation?tex=%20%5Cepsilon%20" alt=" \epsilon " class="ee_img tr_noresize" eeimg="1"> 是偏移量

二、 定义损失函数

<img src="https://www.zhihu.com/equation?tex=%5Ctext%7BL%28%24%5Cbeta_0%24%20%2C%20%24%5Cbeta_1%24%29%7D%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%28y_i%20-%20%28%5Cbeta_0%20%2B%20%5Cbeta_1%20x_i%29%29%20%5E2%7D%5C%5C" alt="\text{L($\beta_0$ , $\beta_1$)} = \frac{1}{N} \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i)) ^2}\\" class="ee_img tr_noresize" eeimg="1">

其中:

<img src="https://www.zhihu.com/equation?tex=%20y_i%20" alt=" y_i " class="ee_img tr_noresize" eeimg="1"> 是第i个样本的真实值

<img src="https://www.zhihu.com/equation?tex=%20x_i%20" alt=" x_i " class="ee_img tr_noresize" eeimg="1"> 是第i个样本的输入

三、 损失最小化

对 <img src="https://www.zhihu.com/equation?tex=%20%5Cbeta_0%20" alt=" \beta_0 " class="ee_img tr_noresize" eeimg="1"> 求导：

<img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cbeta_0%7D%20%3D%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cbeta_0%7D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%28y_i%20-%20%28%5Cbeta_0%20%2B%20%5Cbeta_1%20x_i%29%29%20%5E2%7D%7D%7BN%7D%5C%5C" alt="\frac{\partial L}{\partial \beta_0} = \frac{\partial}{\partial \beta_0} \frac{\sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i)) ^2}}{N}\\" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=%3D%20-%5Cfrac%7B2%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%28y_i-%5Cbeta_0%20-%20%5Cbeta_1%20x_i%29%7D%5C%5C" alt="= -\frac{2}{N}\sum_{i=1}^{N}{(y_i-\beta_0 - \beta_1 x_i)}\\" class="ee_img tr_noresize" eeimg="1">

对 <img src="https://www.zhihu.com/equation?tex=%5Cbeta_1" alt="\beta_1" class="ee_img tr_noresize" eeimg="1"> 求导：

<img src="https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cbeta_1%7D%20%3D%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cbeta_1%7D%20%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%28y_i%20-%20%28%5Cbeta_0%20%2B%20%5Cbeta_1%20x_i%29%29%20%5E2%7D%7D%7BN%7D%5C%5C" alt="\frac{\partial L}{\partial \beta_1} = \frac{\partial}{\partial \beta_1} \frac{\sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i)) ^2}}{N}\\" class="ee_img tr_noresize" eeimg="1">

<img src="https://www.zhihu.com/equation?tex=%3D%20-%5Cfrac%7B2%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7Bx_i%28y_i-%5Cbeta_0-%5Cbeta_1x_i%29%7D%5C%5C" alt="= -\frac{2}{N}\sum_{i=1}^{N}{x_i(y_i-\beta_0-\beta_1x_i)}\\" class="ee_img tr_noresize" eeimg="1">

初始化两个参数 <img src="https://www.zhihu.com/equation?tex=%5Cbeta_0%20%2C%20%5Cbeta_1" alt="\beta_0 , \beta_1" class="ee_img tr_noresize" eeimg="1"> ,计算MSE，通过梯度下降的方式，不断更新参数。

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



Reference:

