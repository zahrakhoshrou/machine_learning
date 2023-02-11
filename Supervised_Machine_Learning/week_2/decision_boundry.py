import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common_w3 import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)
#
# fig, ax = plt.subplots(1,1,figsize=(4,4))
# # fig, ax = plt.subplots(figsize=(4,4))
# plot_data(X,y,ax)
#
# # axis([xmin, xmax, ymin, ymax])
# ax.axis([0, 4, 0, 3.5])
# ax.set_ylabel('$x_1$')
# ax.set_xlabel('$x_0$')
# plt.show()


# z = np.arange(-10,11)
#
# fig, ax = plt.subplots(1,1, figsize=(5,3))
# ax.plot(z, sigmoid(z), c="b")
#
# ax.set_xlabel("Sigmoid function")
# ax.set_ylabel("sigmoid(z)")
# ax.set_title("z")
#
# # plt.show()
# draw_vthresh(ax,0)

"""From what you've learnt above, you can see that this model predicts  𝑦=1  if  −3+𝑥0+𝑥1>=0 
    Let's see what this looks like graphically.
    We'll start by plotting  −3+𝑥0+𝑥1=0 , which is equivalent to  𝑥1=3−𝑥0 """

x0 = np.arange(0,6)
x1 = 3 - x0
fig, ax = plt.subplots(1,1,figsize=(5,4))

ax.plot(x0, x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()