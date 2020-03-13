# the solution is mulit-D polynomial: u(x) = (x1^2-1)*(x2^2-1)*...*(xd^2-1)
# the problem is Laplace u = f, in the domain

import torch 
from numpy import array, prod, sum, zeros

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

h = 0.001 # step length ot compute derivative

# define the true solution for numpy array (N sampling points of d variables)
def true_solution(x_batch):
    u = prod(x_batch**2-1, 1)
    return u

# the point-wise Du: Input x is a sampling point of d variables ; Output is a numpy vector which means the result of Du(x))
def Du(model,x_batch):
    s = zeros(x_batch.shape[0])
    for i in range(x_batch.shape[1]):  
        ei = zeros(x_batch.shape)
        ei[:,i] = 1
        s = s + (model.predict(x_batch+h*ei)-2*model.predict(x_batch)+model.predict(x_batch-h*ei))/h/h
    return s

# the point-wise Du: Input x is a batch of sampling points of d variables (tensor) ; Output is tensor vector which means the result of Du(x))
def Du_ft(model,tensor_x_batch):
    s = torch.zeros(tensor_x_batch.shape[0])
    for i in range(tensor_x_batch.shape[1]):  
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:,i] = 1
        s = s + (model(tensor_x_batch+h*ei)-2*model(tensor_x_batch)+model(tensor_x_batch-h*ei))/h/h
    return s

# define the right hand function for numpy array (N sampling points of d variables)
def f(x_batch):
    f = 2*prod(x_batch**2-1, 1)*sum(1/(x_batch**2-1), 1)
    return f

# the point-wise Bu for tensor (N sampling points of d variables)
def Bu_ft(model,tensor_x_batch):
    return model(tensor_x_batch)

# define the boundary value g for numpy array (N sampling points of d variables)
def g(x_batch):
    u = zeros((x_batch.shape[0],))
    return u

# the point-wise h0 for numpy array (N sampling points of d variables)
def h0(x_batch):
    return None

# the point-wise h1 for numpy array (N sampling points of d variables)
def h1(x_batch):
    return None

# specify the domain type
def domain_shape():
    return 'cube'

# output the domain parameters
def domain_parameter(d):
    intervals = zeros((d,2))
    for i in range(d):
        intervals[i,:] = array([-1,1])
    return intervals

# If this is a time-dependent problem
def time_dependent_type():
    return 'none'

# output the time interval
def time_interval():
    return None
