import torch.nn as nn
import torch.nn.functional as F
import torch
import p03_layers
from torch.autograd import Variable

input = Variable(torch.randn(5))
print (input)
my_elu = p03_layers.P3ELU(2.5, True)
nn_output = F.elu(input, alpha = 2.5)
my_elu(input)
print (input)
print (nn_output)
