import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
# from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

from Model_representstion import compute_model_output

# 𝑓𝑤,𝑏(𝑥(𝑖))  is our prediction for example  𝑖  using parameters  𝑤,𝑏 .
# (𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2  is the squared difference between the target value and the prediction.
# These differences are summed over all the  𝑚  examples and divided by 2m to produce the cost,  𝐽(𝑤,𝑏) .


def compute_cost(x, y, w, b):

    """
       Computes the cost function for linear regression.

       Args:
         x (ndarray (m,)): Data, m examples
         y (ndarray (m,)): target values
         w,b (scalar)    : model parameters

       Returns
           total_cost (float): The cost of using w,b as the parameters for linear regression
                  to fit the data points in x and y
       """

    f_wb = compute_model_output(x, w, b)
    m = x.shape[0]

    temp = 0
    for i in range(m):
        # print(f_wb[i])
        # print(y[i])
        temp = temp + (f_wb[i] - y[i])**2
    result = (1/(2*m)) * temp
    return result

# x_train = np.array([1.0, 2.0])
# y_train = np.array([300, 500])
# print(compute_cost(x_train, y_train, w=200, b=100))


if __name__ == "__main__":

    x_train = np.array([1.0, 2.0])
    y_train = np.array([300, 500])
    compute_cost(x_train, y_train, w=200, b=100)