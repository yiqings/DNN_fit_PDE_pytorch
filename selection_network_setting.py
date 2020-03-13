# build the neural network to approximate the selection net
import numpy
import torch
from torch import tanh, squeeze, sin, sigmoid
from torch.nn.functional import relu

class selection_network(torch.nn.Module):
    def __init__(self, d, m, maxvalue, minvalue, initial_constant = 'none'):
        super(selection_network, self).__init__()
        self.linear1 = torch.nn.Linear(d,m)
        self.linear2 = torch.nn.Linear(m,int(round(m/2)))
        self.linear3 = torch.nn.Linear(int(round(m/2)),1)
        self.maxvalue = maxvalue
        self.minvalue = minvalue
        if not initial_constant == 'none':
            torch.nn.init.constant_(self.linear3.weight, 0.0)
            torch.nn.init.constant_(self.linear3.bias, -numpy.log((maxvalue-initial_constant)/(initial_constant-minvalue))) 

    def forward(self, tensor_x_batch):
        y = relu(self.linear1(tensor_x_batch))
        y = relu(self.linear2(y))
        y = sigmoid(self.linear3(y))*(self.maxvalue-self.minvalue)+self.minvalue
        return y.squeeze(1)
    
    # to evaluate the solution with numpy array input and output
    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()