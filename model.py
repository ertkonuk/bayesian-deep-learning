import torch
from torch import nn
import torch.nn.functional as F
# . . import the bayesian layers
from modules import BayesianConv2d, BayesianLinear

# . .       . . #
# . . CIFAR . . #
# . .       . . #

# . . Bayesian Convolutional Neural Network
# . . define the network architecture
class BayesianCNNClassifierCIFAR(nn.Module):
    # . . the constructor
    def __init__(self, in_channels=3, num_classes=10, priors=None, lrt=False):
        # . . call the constructor of the parent class
        super(BayesianCNNClassifierCIFAR, self).__init__()

        self.num_classes = num_classes 

        # . . the network architecture
        # . . convolutional layerds for feature engineering
        self.conv1 = nn.Sequential(
                     BayesianConv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 32),
                     BayesianConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 32),
                     nn.MaxPool2d(2)
                    )

        self.conv2 = nn.Sequential(
                     BayesianConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 64),
                     BayesianConv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 64),
                     nn.MaxPool2d(2)
                    )

        self.conv3 = nn.Sequential(
                     BayesianConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 128),
                     BayesianConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 128),
                     nn.MaxPool2d(2)
                    )

        # . . fully connected layers for the steering prediction
        self.linear = nn.Sequential(
                      BayesianLinear(128 * 4 * 4, 1024, priors=priors, lrt=lrt),                      
                      nn.ReLU(),
                      BayesianLinear(1024, num_classes, priors=priors, lrt=lrt)
                    )
        
    # . . forward propagation
    def forward(self, x):
        # . . convolutional layers
        x = self.conv1(x)        
        x = self.conv2(x)        
        x = self.conv3(x)        

        # . . flatten the tensor for fully connected layers
        #x = x.view(x.shape[0], -1)        
        x = torch.flatten(x, start_dim=1)
        
        # . . fully connected layers: the classifier
        x = self.linear(x)

        # . . compute the KL divergence loss
        kl_div = 0.0
        # . . iterate over modules and check if they have the kl_div_loss property
        for module in self.modules():
            if hasattr(module, 'kl_div_loss'):
                kl_div = kl_div + module.kl_div_loss()
        
        return x, kl_div

# . .       . . #
# . . MNIST . . #
# . .       . . #

# . . Bayesian Convolutional Neural Network
# . . define the network architecture
class BayesianCNNClassifierMNIST(nn.Module):
    # . . the constructor
    def __init__(self, in_channels=1, num_classes=10, priors=None, lrt=False):
        # . . call the constructor of the parent class
        super(BayesianCNNClassifierMNIST, self).__init__()

        self.num_classes = num_classes 

        # . . the network architecture
        # . . convolutional layerds for feature engineering
        self.conv1 = nn.Sequential(
                     BayesianConv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 32),
                     BayesianConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 32),
                     nn.MaxPool2d(2)
                    )

        self.conv2 = nn.Sequential(
                     BayesianConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 64),
                     BayesianConv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 64),
                     nn.MaxPool2d(2)
                    )

        self.conv3 = nn.Sequential(
                     BayesianConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 128),
                     BayesianConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, priors=priors, lrt=lrt),
                     nn.ReLU(),
                     nn.BatchNorm2d(num_features = 128),
                     nn.MaxPool2d(2)
                    )

        # . . fully connected layers for the steering prediction

        self.linear = nn.Sequential(
                      BayesianLinear(128 * 3 * 3, 1024, priors=priors, lrt=lrt),                      
                      nn.ReLU(),
                      BayesianLinear(1024, num_classes, priors=priors, lrt=lrt)
                    )
        
    # . . forward propagation
    def forward(self, x):
        # . . convolutional layers
        x = self.conv1(x)        
        x = self.conv2(x)        
        x = self.conv3(x)        

        # . . flatten the tensor for fully connected layers
        #x = x.view(x.shape[0], -1)        
        x = torch.flatten(x, start_dim=1)
        
        # . . fully connected layers: the classifier
        x = self.linear(x)

        # . . compute the KL divergence loss
        kl_div = 0.0
        # . . iterate over modules and check if they have the kl_div_loss property
        for module in self.modules():
            if hasattr(module, 'kl_div_loss'):
                kl_div = kl_div + module.kl_div_loss()
        
        return x, kl_div