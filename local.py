# Desc:
#   Implements a Locality Prior Layer, which encourages neurons that fire together to 
#   be physically located close together. "Close" is determined by the layer topology, 
#   and currently a 1D line segment and 2D plane section are allowed. The 2D section
#   can use multiple metrics, and the wiring cost is a function of this distance. 
#   Results can be visualized in the notebook.
# 
# Author:
#   Sasha Sax, SVL
# 
# Usage:
#   For import only

from   collections import OrderedDict
import math
import numpy as np
import scipy.spatial.distance as distance

import torch
from   torch.autograd import Variable
from   torch.nn import functional as F
from   torch.nn import Module
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class LocalityPriorLinear(nn.Linear):
    def __init__(self, inputDimension, outputDimension, topology='euclidean', weight_fn=np.sqrt):
        super(LocalityPriorLinear, self).__init__(inputDimension, outputDimension)
        # prior = simple_1d_prior(inputDimension, outputDimension)
        prior = prior_2d(inputDimension, outputDimension, topology, weight_fn)
        self.register_buffer('prior', torch.FloatTensor(prior))

    def forward(self, x):
        updated_weights = torch.mul(self.weight, Variable(self.prior, requires_grad=False))
        return F.linear(x, updated_weights, self.bias)

# Helpers for LocalityPriorLinear layer
def make_idx_arr(inputDimension, outputDimension):
    x = np.linspace(0, inputDimension-1, inputDimension)
    y = np.linspace(0, outputDimension-1, outputDimension)
    xv, yv = np.meshgrid(x, y)
    return np.stack([yv, xv], axis=2)


def normalize_prior(prior):
    '''The use of a prior should not affect the overall strength of 
    incoming connections to a neuron. This could deteriorate performance
    because this layer would face additional weight regularization. 
    Therefore, we normalize the prior. '''
    z = np.sum(prior, axis=0) / float(prior.shape[0])
    prior /= z[:,np.newaxis]
    return prior

# Set layer topology
def simple_1d_prior(inputDimension, outputDimension, weight_fn=np.sqrt):
    idxs = make_idx_arr(inputDimension, outputDimension)
    dist = np.abs(idxs[:,:,0] - idxs[:,:,1]) + 1
    prior = 1./weight_fn(dist) #1D prior
    return normalize_prior(prior)

def prior_2d(inputDimension, outputDimension, topology='euclidean', weight_fn=np.sqrt):
    ''' A 2D section with the metric-induced topology. The weight function, w(d) is the 
    a reweighting of the cost of wiring two neurons as a function of their distance. '''
    idxs = make_idx_arr(int(math.sqrt(inputDimension)), int(math.sqrt(outputDimension)))
    long_idxs = np.reshape(idxs, (-1,2))
    dist = distance.cdist(long_idxs, long_idxs, metric=topology) + 1
    prior = 1./weight_fn(dist)
    return prior
    # return normalize_prior(prior)


class SelectiveSequential(nn.Module):
    """ This allows features to be extracted from multiple layers of a network.
    Amended from Francisco Massa's answer at 
    https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/8
    """
    def __init__(self, to_select, modules_dict):
        super(SelectiveSequential, self).__init__()
        self.modules_dict = modules_dict
        for key, module in modules_dict.items():
            self.add_module(key, module)
        self._to_select = to_select

    def forward(self, x):
        lst = []
        for name, module in self._modules.iteritems():
            x = module(x)
            if name in self._to_select:
                lst.append(x)
        return lst



""" 
----------------------------MODELS-------------------------
Implements AlexNet with a LocalityPriorLinear layer.

"""
__all__ = ['AlexNet', 'alexnet']
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):
    # Amended from http://pytorch.org/docs/torchvision/models.html
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = SelectiveSequential(
            ['fc5', 'fc6', 'fc7'],
            OrderedDict([
                ('d4', nn.Dropout()),
                ('fc5_', nn.Linear(256 * 6 * 6, 4096)),
                ('fc5', nn.ReLU(inplace=True)),
                ('d5', nn.Dropout()),
                ('fc6_', nn.LocalityPriorLinear(4096, 4096)),
                ('fc6', nn.ReLU(inplace=True)),
                ('fc7', nn.Linear(4096, num_classes))
            ])
        )

    def forward(self, x, return_activations=False):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if return_activations:
            return x
        else:
            return x[-1]


def alexnet_local(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Amended from http://pytorch.org/docs/torchvision/models.html
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model




class MNISTNet(nn.Module):
    def __init__(self, use_local=False):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.use_local = use_local
        if use_local:
            self.fc15 = LocalityPriorLinear(49, 49, weight_fn=lambda x: x)
        else:
            self.fc15 = nn.Linear(49, 49)

        self.classifier = SelectiveSequential(
            ['fc1', 'fc15', 'active'],
            OrderedDict([
                ('d1', nn.Dropout()),
                ('l1', nn.Linear(320, 49)),
                ('fc1', nn.ReLU(inplace=True)),
                ('d2', nn.Dropout()),
                ('fc15_', self.fc15),
                ('fc15', nn.ReLU(inplace=True)),
                ('active', nn.Linear(49, 10))
            ])
        )

    def forward(self, x, return_activations=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.classifier(x)
        if return_activations:
            return x
        else:
            return F.log_softmax(x[-1])

