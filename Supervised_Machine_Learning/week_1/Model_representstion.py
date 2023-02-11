import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300, 500])

m = x_train.shape[0]
# print(m)

for i in range(len(x_train)):
    x_i = x_train[i]
    y_i = y_train[i]
    print(x_i)
    print(y_i)

plt.scatter(x_train, y_train, marker='x')
plt.title("Housing price")
plt.xlabel("size")
plt.ylabel("Price")
# plt.show()

def compute_model_output(x, w, b):
    """
        Computes the prediction of a linear model
        Args:
          x (ndarray (m,)): Data, m examples
          w,b (scalar)    : model parameters
        Returns
          y (ndarray (m,)): target values
        """
    m = x_train.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    return f_wb

# print(compute_model_output(x_train, 100, 100))
w=200
b=100
tmp_f_wb = compute_model_output(x_train,w,b)
# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
# A legend is an area describing the elements of the graph.
# In the matplotlib library, there’s a function called legend()
#   which is used to Place a legend on the axes.
plt.legend()
plt.show()
