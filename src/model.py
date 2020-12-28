import torch
from torch import nn



class ConvBlock(nn.Module):
    """
    A simple convolution block with
    the standard batch_norm + ReLU + maxxpool
    """ 

    activations = {
        "relu": nn.ReLU,
        "elu": nn.ELU
    }
    ops_dim = {
        "2D": {
            "conv": nn.Conv2d, 
            "norm": nn.BatchNorm2d, 
            "pool": nn.MaxPool2d
            },
        "3D": {
            "conv": nn.Conv3d, 
            "norm": nn.BatchNorm3d, 
            "pool": nn.MaxPool3d
            }
    }
    def __init__(self, input_planes, output_planes, kernel,
                 stride, activation="relu", conv_type="2D", use_max_pool=True):
        super(ConvBlock, self).__init__()
        
        self.conv_layer = ConvBlock.ops_dim[conv_type]["conv"](input_planes, 
                                                               output_planes, 
                                                               kernel_size=kernel, 
                                                               stride=stride)
        self.batch_norm = ConvBlock.ops_dim[conv_type]["norm"](output_planes)
        self.activation = ConvBlock.activations[activation]()
        self.use_max_pool= use_max_pool
        if self.use_max_pool:
            if conv_type =="3D":

                self.max_pool = ConvBlock.ops_dim[conv_type]["pool"]((1,2,2))
            else:
                self.max_pool = ConvBlock.ops_dim[conv_type]["pool"](2)
            

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        if self.use_max_pool:
            x = self.max_pool(x)
        return x



class ParallelConvolution(nn.Module):
    """
    Creates a "parallel" convolutional block
    Meant for two adjacent sequences
    """
    def __init__(self, input_planes, 
                 output_planes, kernel_size, stride, conv_type="2D"):
        super(ParallelConvolution, self).__init__()
        self.conv_1 = ConvBlock(input_planes, 
                                output_planes, 
                                kernel_size, 
                                stride,
                                conv_type=conv_type)
        self.conv_2 = ConvBlock(input_planes, 
                                output_planes, 
                                kernel_size, 
                                stride,
                                conv_type=conv_type)
        
    def forward(self, x, x2):
        x = self.conv_1(x)
        x2 = self.conv_2(x2)
        return x, x2



class ParallelFC(nn.Module):
    """
    Creates a "parallel" fully connected layer
    Meant for two adjacent sequences
    """
    def __init__(self, input_dim, output_dim, concat=False):
        super(ParallelFC, self).__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.linear_2 = nn.Linear(input_dim, output_dim)
        self.concat = concat

    def forward(self, x1, x2):
        x1 = self.linear_1(x1)
        x2 = self.linear_2(x2)

        if self.concat:
            x = torch.cat((x1, x2),  1)
            return x
        else:
            return x1, x2

class ParallelConvolution(nn.Module):
    """
    Creates a "parallel" convolutional block
    Meant for two adjacent sequences
    """
    def __init__(self, input_planes, 
                 output_planes, kernel_size, stride, conv_type="2D"):
        super(ParallelConvolution, self).__init__()
        self.conv_1 = ConvBlock(input_planes, 
                                output_planes, 
                                kernel_size, 
                                stride,
                                conv_type=conv_type)
        self.conv_2 = ConvBlock(input_planes, 
                                output_planes, 
                                kernel_size, 
                                stride,
                                conv_type=conv_type)
        
    def forward(self, x, x2):
        x = self.conv_1(x)
        x2 = self.conv_2(x2)
        return x, x2



class VelocityEstimationNet(nn.Module):
    """
    The goal with this architecture is to
    capture the motion between adjacent frames with
    the convolution operations

    Then, the speed is estimated using a concatenation
    of two parallel convnets, each of which has fully
    connected layers. The output of each convnet is
    concatenated prior to additional fully
    connected layers then to a final prediction
    of the speed
    """
    def __init__(self, input_planes):
        super(VelocityEstimationNet, self).__init__()
        self.layer_1 = ParallelConvolution(input_planes, 96, 11, 4)
        self.layer_2 = ParallelConvolution(96, 256, 5, 1)
        self.layer_3 = ParallelConvolution(256, 384, 3, 1)
        self.layer_4 = ParallelConvolution(384, 384, 3, 1)
        self.layer_5 = ParallelConvolution(384, 384, 3, 1)
        self.fc1 = ParallelFC(384, 7680)
        self.fc2 = ParallelFC(7680, 3840, concat=True)
        self.joined_fc1 = nn.Linear(7680, 3840)
        self.joined_fc2 = nn.Linear(3840, 3840)
        self.joined_fc3 = nn.Linear(3840,1)
        

    def forward(self, x1, x2):
        x1, x2 = self.layer_1(x1, x2)
        x1, x2 = self.layer_2(x1, x2)
        x1, x2 = self.layer_3(x1, x2)
        x1, x2 = self.layer_4(x1, x2)
        x1, x2 = self.layer_5(x1, x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x1, x2 = self.fc1(x1, x2)
        
        x = self.fc2(x1, x2)
        x = self.joined_fc1(x)
        x = self.joined_fc2(x)
        x = self.joined_fc3(x)
        return(x)


class DepthVelocityEstimationNet(nn.Module):
    """
    The goal with this architecture is to
    capture the motion between adjacent frames with
    the convolution operations

    Then, the speed is estimated using a concatenation
    of two parallel convnets, each of which has fully
    connected layers. The output of each convnet is
    concatenated prior to additional fully
    connected layers then to a final prediction
    of the speed
    """
    def __init__(self, input_planes, depth):
        super(DepthVelocityEstimationNet, self).__init__()
        self.layer_1 = ParallelConvolution(input_planes=depth, 
                                           output_planes=96, 
                                           stride=(3,4,4), 
                                           conv_type="3D", kernel_size=(3, 11, 11))
        self.layer_2 = ParallelConvolution(96, 256, 5, 1)
        self.layer_3 = ParallelConvolution(256, 384, 3, 1)
        self.layer_4 = ParallelConvolution(384, 384, 3, 1)
        self.layer_5 = ParallelConvolution(384, 384, 3, 1)
        self.fc1 = ParallelFC(384, 7680)
        self.fc2 = ParallelFC(7680, 3840, concat=True)
        self.joined_fc1 = nn.Linear(7680, 3840)
        self.joined_fc2 = nn.Linear(3840, 3840)
        self.joined_fc3 = nn.Linear(3840,depth)
        

    def forward(self, x1, x2):
        x1, x2 = self.layer_1(x1, x2)
        x1 = x1.squeeze(2)
        x2 = x2.squeeze(2)

        x1, x2 = self.layer_2(x1, x2)
        x1, x2 = self.layer_3(x1, x2)
        x1, x2 = self.layer_4(x1, x2)
        x1, x2 = self.layer_5(x1, x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x1, x2 = self.fc1(x1, x2)
        x = self.fc2(x1, x2)
        x = self.joined_fc1(x)
        x = self.joined_fc2(x)
        x = self.joined_fc3(x)
        return(x)


class OpticalFlowConvolutionalNetwork(nn.Module):
    def __init__(self, input_planes=3, dropout=0.2, output_dim=1):
        super(OpticalFlowConvolutionalNetwork, self).__init__()

        
        self.layer_1 = ConvBlock(input_planes=input_planes,
                                 output_planes=96,
                                 kernel=11, 
                                 stride=4,
                                 conv_type="2D",
                                 activation="elu")
        self.layer_2 = ConvBlock(input_planes=96,
                                 output_planes=256,
                                 kernel=5, 
                                 stride=2,
                                 conv_type="2D",
                                 activation="elu")
        self.layer_3 = ConvBlock(input_planes=256,
                                 output_planes=384,
                                 kernel=5, 
                                 stride=2,
                                 conv_type="2D",
                                 activation="elu")
        self.layer_4 = ConvBlock(input_planes=384,
                                 output_planes=384,
                                 kernel=3,
                                 stride=2, 
                                 conv_type="2D",
                                 use_max_pool=False,
                                 activation="elu")
        self.layer_5 = ConvBlock(input_planes=384,
                                 output_planes=384,
                                 kernel=3,
                                 stride=2, 
                                 conv_type="2D",
                                 use_max_pool=False,
                                 activation="elu")
        self.layer_6 = nn.Linear(14976,1000)
        self.layer_7 = nn.Linear(1000, 100)
        self.layer_8 = nn.Linear(100, 10)
        self.layer_9 = nn.Linear(10, 1)

        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = torch.flatten(x, 1)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)

        return x
