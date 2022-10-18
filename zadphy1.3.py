
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5, 5, 0.01)

#f1
y = np.tanh(x)
plt.plot(x, y)

#f2
y = ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))
plt.plot(x, y)


#f3
y =(1/(1+np.exp(-x)))
plt.plot(x, y)

#f4
x = np.arange(-5, 5, 0.01)
y = np.where(x <= 0, 0, x)
plt.plot(x, y)

#f5
plt.plot(x[x>0], x[x>0])
plt.plot(x[x<=0], np.exp(x[x<=0])-1)


