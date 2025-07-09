# 支持向量机
SVM（支持向量机）是一种简单的线性分类模型，当我们输入一组样本，样本中含有两个类别，每一个数据包括了n个特征，我们认为，同意类的数据包含了相似的特征，在坐标系中，应该团成一团。因此，我们希望可以通过一个分割平面（广义的平面，实际上是一种分割的依据，在二维空间是一条直线，三维空间是一个平面，更高的维数只能通过坐标来表示了），准确地将两个类别分开。

## 实验过程
一、 定义直线

$$
y = w x + b 
$$

注意：
这里的x，y与样本中的数据点无关，只是为了表示直线。

二、 定义损失函数

$$
L(w,b) = \max(0 , 1 - y_i(w x_i + b))
$$


其中：

$$
y_i \text{是}x_i\text{的标签}
$$

解释：
根据点到平面的距离公式可知，为了让点到平面的距离足够大（如果不够大，可能导致分类的模糊），因此我们要求：

$$
|w x_i + b| \ge 1
$$

Q：为什么取1? 

A：方便计算，可以取其他的数，但是可以通过w和b的缩放来控制。

$$
\text{当} |w x_i + b| < 1 \text{的时候，我们认为，这时候的预测不够准确，而其他的情况下，预测足够准确了，所以得到了上述的损失函数。}
$$

三、 损失函数最小化

$$
\frac{\partial L}{\partial w} = \frac{1}{N} \sum_{i=1}^{N}(-y_i x_i) 
$$

$$
\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N}(y_i \cdot sgn(1 - y_i(w x_i + b))) 
$$

$$
f(x) = 
\begin{cases}
1,& x \ge 1 \\
-1& x < 1
\end{cases}
$$

```python
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

```

