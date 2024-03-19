import torch.nn.functional as F
from min_max import *

class AttentionModel(nn.Module):

    def __init__(self, num_classes, filters_in, filters_out, dropout):

        super(AttentionModel, self).__init__()

        self.filters_out = filters_out
        self.filters_in = filters_in
        self.dropout = dropout
        self.num_classes = num_classes

        # 3D convolution of the tensor
        self.conv_filter = nn.Conv3d(in_channels=1, out_channels=self.filters_out, kernel_size=(512, 3, 3), padding=(0, 1, 1))

        # Computation of max and min pool layer
        self.min_max = MinMaxLayer(kernel_size=2)

        # Linear layer for the classification
        self.linear = nn.Linear(in_features=self.filters_in*self.filters_out*2, out_features=num_classes)

        # ReLU activation function for the feature extraction
        self.relu = nn.ReLU(inplace=True)

        # Dropout layer
        self.drop_out = nn.Dropout(dropout)


    def forward(self, x):

        # 3D convolution computation
        x_conv = self.conv_filter(x)

        # Squeeze the tensor to make it 3D
        x_conv = torch.squeeze(x_conv)

        # Max and Min Pooling
        x_max, x_min = self.min_max(x_conv)

        # Softmax of the Attention Map
        x_max = F.softmax(x_max.view((x_max.shape[1], x_max.shape[2] * x_max.shape[3])), dim=1).view(x_max.shape)
        x_min = F.softmax(x_min.view((x_min.shape[1], x_min.shape[2] * x_min.shape[3])), dim=1).view(x_min.shape)

        # Unsqueeze the tensors for the product computation
        x_max = x_max.unsqueeze(2)
        x_min = x_min.unsqueeze(2)

        # Product computation between the Attention Maps and the input tensor
        v1 = self.relu(self._featureExtraction(x, x_max))
        v2 = self.relu(self._featureExtraction(x, x_min))
        
        # Concatenation of the two vectors
        x4=torch.cat((v1, v2), 0)

        # Linear layer for the classification
        output = self.drop_out(self.linear(x4))

        return output


    def _featureExtraction(self, x, map):

        #reshape of tensors for the computation of the product between map and input tensor x
        map = map.reshape((map.shape[1], map.shape[0], map.shape[2], map.shape[3], map.shape[4]))
        x = x.reshape((x.shape[1], x.shape[0], x.shape[2], x.shape[3], x.shape[4]))

        ris = F.conv3d(input=x, weight=map, bias=None, stride=1, padding=0, groups=1)

        ris = torch.squeeze(ris)
        ris = ris.reshape((ris.shape[0] * ris.shape[1]))

        return ris
    