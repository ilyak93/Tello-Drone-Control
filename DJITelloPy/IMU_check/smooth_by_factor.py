import numpy as np
import matplotlib.pyplot as plt

x_dist = np.load("x_dist.npy")
y_dist = np.load("y_dist.npy")
z_dist = np.load("z_dist.npy")

print(x_dist.mean())
print(y_dist.mean())
print(z_dist.mean())


x_axis = list(range(1, len(x_dist)+1))
y_axis = x_dist
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('x-dist')
plt.title("x_dist_from_pad")
plt.show()


x_axis = list(range(1, len(y_dist)+1))
y_axis = y_dist
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('y-dist')
plt.title("y_dist_from_pad")
plt.show()

x_axis = list(range(1, len(z_dist)+1))
y_axis = z_dist
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('z-dist')
plt.title("z_dist_from_pad")
plt.show()

n = len(x_dist)
smooth_elements = 100
x_dist_smoothed = np.average(x_dist.reshape(-1, smooth_elements), axis=1)
y_dist_smoothed = np.average(y_dist.reshape(-1, smooth_elements), axis=1)
z_dist_smoothed = np.average(z_dist.reshape(-1, smooth_elements), axis=1)




x_axis = list(range(1, len(x_dist_smoothed)+1))
y_axis = x_dist_smoothed
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('x-dist')
plt.title("x_dist_from_pad_smoothed_by_factor_10")
plt.show()


x_axis = list(range(1, len(y_dist_smoothed)+1))
y_axis = y_dist_smoothed
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('y-dist')
plt.title("y_dist_from_pad_smoothed_by_factor_10")
plt.show()


x_axis = list(range(1, len(z_dist_smoothed)+1))
y_axis = z_dist_smoothed
plt.plot(x_axis, y_axis)
plt.xlabel('experiment #')
plt.ylabel('z-dist')
plt.title("z_dist_from_pad_smoothed_by_factor_10")
plt.show()