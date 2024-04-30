import torch
from typing import Callable

class Transpose(Callable): 

    def __init__(self, dim0 : int, dim1 : int):
        self.dim0 = dim0
        self.dim1 = dim1

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, dim0=self.dim0, dim1=self.dim1)
    
class Flip(Callable):

    def __init__(self, dims : tuple[int,...]):
        self.dims = dims

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=self.dims)
    
class DownShift(Callable):

    def __init__(self,shift : int):
        self.shift = shift

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        
        h, w, c = x.shape

        y = torch.zeros(x.shape)
        y[self.shift:h,:,:] = x[:h-self.shift,:,:]

        return y
    

class UpShift(Callable):

    def __init__(self,shift : int):
        self.shift = shift

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        
        h, w, c = x.shape

        y = torch.zeros(x.shape)
        y[:h-self.shift,:,:] = x[self.shift:h,:,:]

        return y
    
class LeftShift(Callable):

    def __init__(self,shift : int):
        self.shift = shift

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        
        h, w, c = x.shape

        y = torch.zeros(x.shape)
        y[:,self.shift:w,:] = x[:,:w - self.shift,:]

        return y
    

class RightShift(Callable):

    def __init__(self,shift : int):
        self.shift = shift

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        
        h, w, c = x.shape

        y = torch.zeros(x.shape)
        y[:,:w - self.shift,:] = x[:,self.shift:w,:]

        return y