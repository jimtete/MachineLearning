import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

x = np.random.normal(size=500)
y = np.random.normal(size=500)
z = np.random.normal(size=500)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x,y,z, c=np.linalg.norm([x,y,z],axis=0))
