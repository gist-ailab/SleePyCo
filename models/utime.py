import torch
import torch.nn as nn
from .utils import Conv1d


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(ConvUnit, self).__init__()
        
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
            )
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class UTimeEncoder(nn.Module):
    def __init__(self, config):
        super(UTimeEncoder, self).__init__()
        
        torch.backends.cudnn.deterministic = False
        
        self.training_mode = config['training_params']['mode']

        self.conv1_1 = ConvUnit(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding='SAME', dilation=2)
        self.conv1_2 = ConvUnit(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding='SAME', dilation=2)
        self.mp1 = nn.MaxPool1d(kernel_size=8)
        self.conv2_1 = ConvUnit(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding='SAME', dilation=2)
        self.conv2_2 = ConvUnit(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='SAME', dilation=2)
        self.mp2 = nn.MaxPool1d(kernel_size=6)
        self.conv3_1 = ConvUnit(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='SAME', dilation=2)
        self.conv3_2 = ConvUnit(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='SAME', dilation=2)
        self.mp3 = nn.MaxPool1d(kernel_size=4)
        self.conv4_1 = ConvUnit(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding='SAME', dilation=2)
        self.conv4_2 = ConvUnit(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding='SAME', dilation=2)
        self.mp4 = nn.MaxPool1d(kernel_size=2)
        self.conv5_1 = ConvUnit(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding='SAME', dilation=2)
        self.conv5_2 = ConvUnit(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding='SAME', dilation=2)

        if self.training_mode == 'freezefinetune' or self.training_mode == 'scratch':
            self.fp_dim = config['feature_pyramid']['dim']
            self.num_scales = config['feature_pyramid']['num_scales']
            self.conv_c5 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)

            if self.num_scales > 1:
                self.conv_c4 = nn.Conv1d(128, self.fp_dim, 1, 1, 0)
            
            if self.num_scales > 2:
                self.conv_c3 = nn.Conv1d(64, self.fp_dim, 1, 1, 0)

        if config['backbone']['init_weights']:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = []
        
        x = self.conv1_1(x)
        c1 = self.conv1_2(x)
        
        x = self.mp1(c1)
        x = self.conv2_1(x)
        c2 = self.conv2_2(x)
        
        x = self.mp2(c2)
        x = self.conv3_1(x)
        c3 = self.conv3_2(x)
        
        x = self.mp3(c3)
        x = self.conv4_1(x)
        c4 = self.conv4_2(x)
        
        x = self.mp4(c4)
        x = self.conv5_1(x)
        c5 = self.conv5_2(x)

        if self.training_mode == 'pretrain':
            out.append(c5)
        elif self.training_mode in ['scratch', 'fullyfinetune', 'freezefinetune']:
            p5 = self.conv_c5(c5)
            out.append(p5)
            if self.num_scales > 1:
                p4 = self.conv_c4(c4)
                out.append(p4)
            if self.num_scales > 2:
                p3 = self.conv_c3(c3)
                out.append(p3)
        
        return out
