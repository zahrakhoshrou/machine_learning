import numpy as np
import time

#
# # NumPy routines which allocate memory and fill arrays with value
# a = np.zeros(4)
# print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.zeros((4,))
# print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.random.random_sample(4)
# print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# b = np.random.random_sample((4,3))
# print(f"np.random.rand(4,3): b = {b}, b shape = {b.shape}, b data type = {b.dtype}")
#
# # Some data creation routines do not take a shape tuple:
# # NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
# a = np.arange(4.)
# print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# a = np.random.rand(4)
# print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
# b = np.random.rand(4,3)
# print(f"np.random.rand(4,3): b = {b}, b shape = {b.shape}, b data type = {b.dtype}")
#
# # values can be specified manually as well:
# # These have all created a one-dimensional vector a with four elements. a.shape returns the dimensions.
# #   Here we see a.shape = (4,) indicating a 1-d array with 4 elements.
# # NumPy routines which allocate memory and fill with user specified values
# a = np.array([5, 4, 3, 2])
# print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
# a = np.array([5., 4, 3, 2])
# print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
#
# # Exercise myself
#
# a = np.zeros(10)
# print(a)
# a = np.zeros((10,10))
# print(a)
# a = np.array([[1,2],[2,3],[4,5]])
# print(a)
# a = np.arange(10)
# print(a)

# # Operations on Vectors
# # Elements of vectors can be accessed via indexing and slicing.
# # Indexing means referring to an element of an array by its position within the array:
# # vector indexing operations on 1-D vectors
# a = np.arange(10)
# print(a)
#
# # access an element
# print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")
#
# # access the last element, negative indexes count from the end
# print(f"a[-1] = {a[-1]}")
#
# # indexs must be within the range of the vector or they will produce and error
# try:
#     c = a[10]
# except Exception as e:
#     print("The error message you'll see is:")
#     print(e)
#
# # Slicing means getting a subset of elements from an array based on their indices.
# # Slicing creates an array of indices using a set of three values (start:stop:step)
#
# # vector slicing operations
# a = np.arange(10)
# print(f"a         = {a}")
#
# # access 5 consecutive elements (start:stop:step)
#
# c = a[2:7:1]
# print("a[2:7:1] = ", c)
#
# # There are a number of useful operations that involve operations on a single vector.
#
# a = np.array([1,2,3,4])
# print(f"a             : {a}")
# # negate elements of a
# b = -a
# print(f"b = -a        : {b}")
#
# # sum all elements of a, returns a scalar
# b = np.sum(a)
# print(f"b = np.sum(a) : {b}")
#
# b = np.mean(a)
# print(f"b = np.mean(a): {b}")
#
# b = a**2
# print(f"b = a**2      : {b}")
#
# a = np.array([1,2,3,4])
# print(f"a             : {a}")
# # negate elements of a
# b = -a
# print(f"b = -a        : {b}")
#
# # sum all elements of a, returns a scalar
#
# b = np.sum(a)
# print(f"b = np.sum(a) : {b}")
# b = np.mean(a)
# print(f"b = np.mean(a): {b}")
# b = a**2
# print(f"b = a**2      : {b}")
#
# # access 3 elements separated by two
# c = a[2:7:2]
# print("a[2:7:2] = ", c)
#
# # access all elements index 3 and above
# c = a[3:]
# print("a[3:]    = ", c)
#
# # access all elements below index 3
# c = a[:3]
# print("a[:3]    = ", c)
#
# # access all elements
# c = a[:]
# print("a[:]     = ", c)

# Vector Vector element-wise operations
# Most of the NumPy arithmetic, logical and comparison operations apply to vectors as well.
#   These operators work on an element-by-element basis. For example
# 𝑐𝑖=𝑎𝑖+𝑏𝑖

# a = np.array([1, 2, 3, 4])
# b = np.array([-1,-2, 3, 4])
# print(f"Binary operators work element wise: {a + b}")
# # try a mismatched vector operation
# c = np.array([1, 2])
# try:
#     d = a + c
# except Exception as e:
#     print("The error message you'll see is:")
#     print(e)
#
# # Scalar Vector operations
# a = np.array([1, 2, 3, 4])
# # multiply a by a scalar
# b = 5 * a
# print(f"b = 5 * a : {b}")


# def my_dot(a, b):
#     """
#    Compute the dot product of two vectors
#
#     Args:
#       a (ndarray (n,)):  input vector
#       b (ndarray (n,)):  input vector with same dimension as a
#
#     Returns:
#       x (scalar):
#     """
#     x = 0
#     for i in range(a.shape[0]):
#         x = x + a[i]*b[i]
#     return x
#
# a = np.array([1,2])
# b = np.array([3,4])
#
# print(my_dot(a, b))
#
# np.random.seed(1)
# a = np.random.rand(10000000)  # very large arrays
# b = np.random.rand(10000000)
#
# tic = time.time()  # capture start time
# c = np.dot(a, b)
# toc = time.time()  # capture end time
#
# print(f"np.dot(a, b) =  {c:.4f}")
# print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")
#
# tic = time.time()  # capture start time
# c = my_dot(a,b)
# toc = time.time()  # capture end time
#
# print(f"my_dot(a, b) =  {c:.4f}")
# print(f"loop version duration: {1000*(toc-tic):.4f} ms ")
#
# del(a);del(b)  #remove these big arrays from memory

# # Matrices

# a = np.zeros((3, 5))
# print(a)
# b = np.random.random_sample((2,3))
# print(b)
# c = np.array([[1,2],[3,4]])
# print(c)
# print(c.shape[0])
# print(c.shape)

# 4.4 Operations on Matrices
# Let's explore some operations using matrices.
#
#
# 4.4.1 Indexing
# Matrices include a second index. The two indexes describe [row, column].
# Access can either return an element or a row/column. See below:

# vector indexing operations on matrices
# a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
# print(f"a.shape: {a.shape}, \na= {a}")
#
# #access an element
# print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")
#
# #access a row
# print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

#vector 2-D slicing operations
a= np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")