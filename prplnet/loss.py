"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""
import numpy as np

from prplnet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    """
    MSE is mean squared error, although we're
    just going to do total squared error
    """
    #TODO: #1 Change this to Mean Squared Error
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

#TODO: Implement cross entropy and other classic loss functions
#TODO: Add in some kind of regularization term to penalize larger weights (you'd have to change API)