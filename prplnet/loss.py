"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""
#TODO: #6 Implement cross entropy and other classic loss functions
#TODO: #5 Add in some kind of regularization term to penalize larger weights (you'd have to change API)
import numpy as np

from prplnet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    """
    MSE is mean squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean((predicted - actual)**2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

