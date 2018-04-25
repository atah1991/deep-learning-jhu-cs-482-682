import torch.nn as nn
import torch
import p03_layers
from torch.autograd import Variable

input = torch.randn(5)
print (input)
m = p03_layers.P3Dropout(p=0.5, inplace = False, training = True)
output = m(Variable(input))

print (input)
print (output)
print (output.data.nonzero())
