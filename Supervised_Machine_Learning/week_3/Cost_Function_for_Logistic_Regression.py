import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common_w3 import  plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])    #(m,)
#
# fig, ax = plt.subplots(1, 1, figsize=(4,4))
# plot_data(X_train, y_train, ax)
#
# # Set both axes to be from 0-4
# ax.axis([0, 4, 0, 3.5])
# ax.set_ylabel('$x_1$', fontsize=12)
# ax.set_xlabel('$x_0$', fontsize=12)
# plt.show()


def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z = np.dot(w,X[i]) + b
        f_wb = sigmoid(z)
        loss = -y[i] * np.log(f_wb) - (1-y[i]) * np.log(1-f_wb)
        cost += loss
    cost = cost / m
    return cost

w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))

"""Now, let's see what the cost function output is for a different value of  𝑤 .

In a previous lab, you plotted the decision boundary for  𝑏=−3,𝑤0=1,𝑤1=1 . That is, you had b = -3, w = np.array([1,1]).

Let's say you want to see if  𝑏=−4,𝑤0=1,𝑤1=1 , or b = -4, w = np.array([1,1]) provides a better model.

Let's first plot the decision boundary for these two different  𝑏  values to see which one fits the data better.

For  𝑏=−3,𝑤0=1,𝑤1=1 , we'll plot  −3+𝑥0+𝑥1=0  (shown in blue)
For  𝑏=−4,𝑤0=1,𝑤1=1 , we'll plot  −4+𝑥0+𝑥1=0  (shown in magenta)"""
# Choose values between 0 and 6
x0 = np.arange(0,6)

# Plot the two decision boundaries
x1 = 3 - x0
x1_other = 4 - x0

fig,ax = plt.subplots(1, 1, figsize=(4,4))
# Plot the decision boundary
ax.plot(x0,x1, c=dlc["dlblue"], label="$b$=-3")
ax.plot(x0,x1_other, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()


