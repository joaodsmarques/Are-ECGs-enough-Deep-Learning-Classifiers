#Code adapted from https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html

import torch
import torch.nn as nn


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

#Base class for VGG family

class VGG(nn.Module):

    def __init__(self, features, in_channels, out_classes, signal_length, p_dropout,):
        super(VGG, self).__init__()

        self.signal_len = signal_length
        self.in_channels = in_channels

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(7)

        #gets the output size of the feature space
        flattened_size = self.calculate_flattened_size()

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(True),
            nn.Dropout(p_dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p_dropout),
            nn.Linear(4096, out_classes),
        )

    def calculate_flattened_size(self):
        """Helper function to compute the flattened feature size after self.features."""
        # It makes a dummy 
        with torch.no_grad():
            # Create a dummy input with the given signal length and channels
            dummy_input = torch.zeros(1, self.in_channels, self.signal_len)
            features_output = self.features(dummy_input)
            features_output = self.avgpool(features_output)
            return features_output.view(1, -1).size(1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels = 12):
    layers = []
    #ECGs by default have 12 leads
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

#Configurations for each VGG architecture
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg, batch_norm, in_channels, out_classes, signal_length, p_dropout,**kwargs):
    
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, in_channels = in_channels), in_channels, out_classes, signal_length, p_dropout, **kwargs)
    
    return model


def vgg11(in_channels, out_classes, signal_length, p_dropout, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('A', False,in_channels, out_classes, signal_length, p_dropout, **kwargs)



def vgg11_bn(in_channels, out_classes, signal_length, p_dropout, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('A',True,in_channels, out_classes, signal_length, p_dropout, **kwargs)



def vgg13(in_channels, out_classes, signal_length, p_dropout, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('B',False, in_channels, out_classes, signal_length, p_dropout, **kwargs)



def vgg13_bn(in_channels, out_classes, signal_length, p_dropout, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('B', True, in_channels, out_classes, signal_length, p_dropout,**kwargs)



def vgg16(in_channels, out_classes, signal_length, p_dropout, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('D', False, in_channels, out_classes, signal_length, p_dropout,**kwargs)



def vgg16_bn(in_channels, out_classes, signal_length, p_dropout, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('D', True, in_channels, out_classes, signal_length, p_dropout,**kwargs)



def vgg19(in_channels, out_classes, signal_length, p_dropout, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('E', False, in_channels, out_classes, signal_length, p_dropout, **kwargs)



def vgg19_bn(in_channels, out_classes, signal_length, p_dropout,  **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('E', True, in_channels, out_classes, signal_length, p_dropout, **kwargs)
