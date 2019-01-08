import os
import sys
import torch
import torch.nn as nn
import math

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


__all__ = ['ResNet', 'resnet18', 'resnet50', 'resnet101'] # resnet101 is coming soon!


model_urls = {
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attlayer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if attlayer == None:
            self.gate = None
        else:
            self.gate = attlayer(planes * 4)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.gate != None:
            out = self.gate(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, attention_type=None):
        self.inplanes = 64 
        super(ResNet, self).__init__()

        if attention_type == None:
            self.attlayer = None
        elif attention_type == 'se':
            from .attention_layers import SELayer as attlayer
            self.attlayer = attlayer
        elif attention_type == 'style_attention':
            from .attention_layers import StyleAttentionLayer as attlayer
            self.attlayer = attlayer
        else:
            raise NotImplementedError
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attlayer=self.attlayer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attlayer=self.attlayer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(attention_type=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], attention_type=attention_type, **kwargs)

    if attention_type == None:
        model_path = 'pretrained_model/resnet50_none/model_best.pth.tar'
    elif attention_type == 'se':
        model_path = 'pretrained_model/resnet50_se/model_best.pth.tar'
    elif attention_type == 'style_attention':
        model_path = 'pretrained_model/resnet50_sa/model_best.pth.tar'
    else:
        raise NotImplementedError

    print("Loading pretrained weights from %s" %(model_path))
    state_dict = torch.load(model_path)['state_dict']
    converted_state_dict = {}

    for name, weight in state_dict.items():
      # Remove 'module.' from key to load weights
      converted_name = name.replace('module.', '')
      converted_state_dict[converted_name] = weight
    del state_dict

    model.load_state_dict(converted_state_dict)

    return model
