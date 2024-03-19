import torch.nn as nn
import torch

class MinMaxLayer(nn.Module):

    def __init__(self, kernel_size):

        super().__init__()
        self._kernel_size = kernel_size

    def forward(self, x):

        # Perform max and min pooling
        x_max = self._max_pool(x)
        x_min = self._min_pool(x)
        
        return x_max, x_min
    
    def _max_pool(self, x):

        MaxPool = nn.MaxPool2d(kernel_size=self._kernel_size, stride=self._kernel_size, return_indices=True)
        MaxUnpool = nn.MaxUnpool2d(kernel_size=self._kernel_size, stride=self._kernel_size)

        # Unsqueeze the input tensor to make it 4D to be compatible with max pooling
        unsqueezed_x = x.unsqueeze(0)

        # Perform max pooling
        x_max, indices = MaxPool(unsqueezed_x.type(torch.FloatTensor))

        # Perform max unpooling
        y = MaxUnpool(x_max, indices,output_size=torch.Size([1, x.shape[0], x.shape[1], x.shape[2]]))

        # Convert the output to float
        y = y.type(torch.FloatTensor)

        return y
    
    def _min_pool(self, x):

        MinPool = nn.MaxPool2d(kernel_size=self._kernel_size, stride=self._kernel_size, return_indices=True)
        MinUnpool = nn.MaxUnpool2d(kernel_size=self._kernel_size, stride=self._kernel_size)

        # Inversion of input
        x = -x

        # Unsqueeze the input tensor to make it 4D to be compatible with max pooling
        unsqueezed_x = x.unsqueeze(0)

        # Perform max pooling on the inverted input to get the minimum pooling
        x_min, indices = MinPool(unsqueezed_x.type(torch.FloatTensor))

        # Perform max unpooling
        y = MinUnpool(x_min, indices, output_size=torch.Size([1, x.shape[0], x.shape[1], x.shape[2]]))

        # Invert the output
        y = -y

        # Convert the output to float
        y = y.type(torch.FloatTensor)

        return y