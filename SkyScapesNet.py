import torch
import torch.nn as nn


class SeparableLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out):
        super().__init__()
        self.add_module('batch_norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('sep_conv', nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                              stride=1, padding=1, bias=True, groups=in_channels))
        self.add_module('drop', nn.Dropout2d(drop_out))


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([SeparableLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            # we pass all previous activations into each dense layer normally
            # But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x
