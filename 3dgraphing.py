import numpy as np
import matplotlib.pyplot as plt

ax = plt.axes(projection="3d")

x_data = np.arange(0,1,0.1)
y_data = np.arange(0,50,0.1)
z_data = np.arange(0.5,50,0.1)

ax.scatter(x_data,y_data,z_data)
plt.show()