# build the neural network to approximate the solution
import torch
from torch import tanh, squeeze, sin, cos, sigmoid, autograd
from torch.nn.functional import relu
import torch.nn as nn

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

# a 2-layer feed forward network for the PDE solution
class network(torch.nn.Module):
    def __init__(self, d, m, activation_type = 'ReLU3', boundary_control_type = 'none',initial_control_type = 'none', flag_zero_initially = False):
        super(network, self).__init__()
        self.fc1 = nn.Linear(d+1, m)
        self.fc2 = nn.Linear(m, m)
        
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)

        self.fc7 = nn.Linear(m, m)
        self.fc8 = nn.Linear(m, m)

        self.fc9 = nn.Linear(m, m)
        self.fc10 = nn.Linear(m, m)

        self.fc11 = nn.Linear(m, m)
        self.fc12 = nn.Linear(m, m)

        self.fc13 = nn.Linear(m, m)
        self.fc14 = nn.Linear(m, m)

        self.fc15 = nn.Linear(m, m)
        self.fc16 = nn.Linear(m, m)

        self.fc17 = nn.Linear(m, m)
        self.fc18 = nn.Linear(m, m)

        self.outlayer = nn.Linear(m, 1,bias = False)

        self.d = d
        self.m = m

        if activation_type == 'ReLU3':
            self.activation = lambda x: relu(x**3)
        elif activation_type == 'sigmoid':
            self.activation = lambda x: sigmoid(x)
        elif activation_type == 'tanh':
            self.activation = lambda x: tanh(x)
        elif activation_type == 'sin':
            self.activation = lambda x: sin(x)
        self.boundary_control_type = boundary_control_type
        if boundary_control_type == 'none':
            self.if_boundary_controlled = False
        else:
            self.if_boundary_controlled = True
        self.initial_control_type = initial_control_type
        if initial_control_type == 'none':
            self.if_initial_controlled = False
        else:
            self.if_initial_controlled = True


        if boundary_control_type == 'none':
            self.if_boundary_controlled = False
        else:
            self.if_boundary_controlled = True

        




        if flag_zero_initially == True:
            torch.nn.init.constant_(self.outlayer.weight, 0.0)
            torch.nn.init.constant_(self.outlayer.bias, 0.0) 


    def forward(self, tensor_x_batch,):
        x = tensor_x_batch
        #s = x#@Ix
        y = self.fc1(x)

        s=y
        
        y = self.activation(y)
        y = self.fc2(y)
        y = self.activation(y)
        y = y/self.m+s
        
        s=y
        y = self.fc3(y)
        y = self.activation(y)
        y = self.fc4(y)
        y = self.activation(y)
        y = y/self.m +s
        
        s=y
        y = self.fc5(y)
        y = self.activation(y)
        y = self.fc6(y)
        y = self.activation(y)
        y = y/self.m +s

        s=y
        y = self.fc7(y)
        y = self.activation(y)
        y = self.fc8(y)
        y = self.activation(y)
        y = y/self.m +s

        s=y
        y = self.fc9(y)
        y = self.activation(y)
        y = self.fc10(y)
        y = self.activation(y)
        y = y/self.m +s

        s=y
        y = self.fc11(y)
        y = self.activation(y)
        y = self.fc12(y)
        y = self.activation(y)
        y = y/self.m +s

        s=y
        y = self.fc13(y)
        y = self.activation(y)
        y = self.fc14(y)
        y = self.activation(y)
        y = y/self.m +s

        s=y
        y = self.fc15(y)
        y = self.activation(y)
        y = self.fc16(y)
        y = self.activation(y)
        y = y/self.m +s

        s=y
        y = self.fc17(y)
        y = self.activation(y)
        y = self.fc18(y)
        y = self.activation(y)
        y = y/self.m +s
        
        y = self.outlayer(y)
        
        y = y.squeeze(1)

        if self.boundary_control_type == 'homo_unit_cube':
            y = torch.prod(tensor_x_batch[:,1:]**2-1, 1)*y
        elif self.boundary_control_type == 'homo_unit_sphere':
            y = (torch.sum(tensor_x_batch[:,1:]**2, 1)-1)*y    
        if self.initial_control_type == 'homo_parabolic':
            y = tensor_x_batch[:,0]*y
        elif self.initial_control_type == 'homo_hyperbolic':
            y = (tensor_x_batch[:,0]**2)*y
        return y
    
    # to evaluate the solution with numpy array input and output
    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    
    # evaluate the second derivative at for k-th spatial coordinate
    def D2_exact(self, tensor_x_batch, k):
        y = self.forward(tensor_x_batch)
        tensor_weight = torch.ones(y.size())
        grad_y = autograd.grad(y, tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True, create_graph=True, only_inputs=True)
        D2y_k = autograd.grad(outputs=grad_y[0][:,k+1], inputs=tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True)[0][:,k+1]
        return D2y_k

    # evaluate the Laplace at tensor_x_batch
    def Laplace(self, tensor_x_batch):
        d = tensor_x_batch.shape[1]-1
        y = self.forward(tensor_x_batch)
        tensor_weight = torch.ones(y.size())
        grad_y = autograd.grad(y, tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True, create_graph=True, only_inputs=True)
        Laplace_y = torch.zeros(y.size())
        for i in range(1,d+1):
            Laplace_y = Laplace_y + autograd.grad(outputs=grad_y[0][:,i], inputs=tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True)[0][:,i]
        return Laplace_y

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    