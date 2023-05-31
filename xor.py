"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np

from prplnet.train import train
from prplnet.nn import NeuralNet
from prplnet.layers import Linear, Tanh

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

targets = np.array([
    [1,0], # [IS_TRUE, IS_FALSE]
    [0,1],
    [0,1],
    [1,0]
])


net = NeuralNet([
    Linear(input_size=2, output_size=2), # Can't be learned by just one linear layer
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    # going to add in argmax for better output
    predicted = np.round(predicted)
    print(x, predicted, y)