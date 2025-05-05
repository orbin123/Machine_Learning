# Gradient Descent for Linear Regression 
# yhat = mx + b
# loss(C) = (y-yhat)**2/N

import numpy as np


# Initiualize x and y
x = np.random.randn(10,1)
y = 2*x + np.random.rand()

print(x)
print(y)

# Initialise some parameters
m = 0.9
b = 0.1

# Hyperparameter
learning_rate = 0.01

# Create gradient descent function
def grad_desc(x, y, m , b, learning_rate ):
    dldm = 0.0 # partial derivative with respect to m
    dldb = 0.0 # partial derivative with respect to b
    N = x.shape[0]
    # loss = (y-(wx+b ))**2
    for xi, yi in zip(x,y):
        dldm += -2*xi*(yi-(m*xi+b))
        dldb += -2*(yi-(m*xi+b))

    # Make an update to the w parameter
    m = m-learning_rate*(1/N)*dldm
    b = b-learning_rate*(1/N)*dldb

    return m,b

# Iteratively make updates
for epoch in range(800):
    m, b = grad_desc(x, y, m, b, learning_rate)
    yhat = m*x+b
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])
    print(f'{epoch} loss is {loss}, parameters w:{m}, b:{b}')

# last iteration 
#loss is [1.09446498e-14], parameters w:[1.99999999], b:[0.97367819]
