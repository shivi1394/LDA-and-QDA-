# LDA-and-QDA-

Homework Question (for the code uploaded):
mport numpy as np

import matplotlib.pyplot as plt
x = np.linspace(0,1,200)
y = np.zeros_like(x,dtype = np.int32)
x[0:100] = np.sin(4*np.pi*x)[0:100]
x[100:200] = np.cos(4*np.pi*x)[100:200]
y = 4*np.linspace(0,1,200)+0.3*np.random.randn(200)
label= np.ones_like(x)
label[0:100]=0
plt.scatter(x,y,c=label)

Run the above python file to generate data. The data sets contain 200 data points and they belong to 2 classes, 
where the first 100 data points are labeled as class 0 and the second data points are labeled as class 1. 
Apply LDA and QDA to data set to find its decision boundary respectively.  
Plot your decision boundaries in your answer sheet as well.  
Note that, when generating y, a random normalized noise is added with a 0.3 factor. 
You can change the value to see how your decision boundary might be affected.
