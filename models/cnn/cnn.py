import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, n_layers: list, maxpool_size: list):
        super(Encoder, self).__init__()
        self.init_layer = EncoderBlock(in_channels=in_channels[0], out_channels=out_channels[0], n_layers=n_layers[0], maxpool_size=maxpool_size[0], first=True)
        self.layer1 = EncoderBlock(in_channels=in_channels[1], out_channels=out_channels[1], n_layers=n_layers[1], maxpool_size=maxpool_size[1])
        self.layer2 = EncoderBlock(in_channels=in_channels[2], out_channels=out_channels[2], n_layers=n_layers[2], maxpool_size=maxpool_size[2])
        self.layer3 = EncoderBlock(in_channels=in_channels[3], out_channels=out_channels[3], n_layers=n_layers[3], maxpool_size=maxpool_size[3])
        self.layer4 = EncoderBlock(in_channels=in_channels[4], out_channels=out_channels[4], n_layers=n_layers[4], maxpool_size=maxpool_size[4])
        
    def forward(self, x: torch.Tensor):
        c1 = self.init_layer(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        return c3, c4, c5 
        
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, maxpool_size, first=False):
        super(EncoderBlock, self).__init__()
        self.first = first
        self.pool = MaxPool1d(maxpool_size)
        self.layers = self.make_layers(in_channels, out_channels, n_layers)
        self.prelu = nn.PReLU()
    
    def make_layers(self, in_channels, out_channels, n_layers):
        layers = []
        for i in range(n_layers):
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            layers += [conv1d, nn.BatchNorm1d(out_channels)]
            if i == n_layers - 1:
                self.gate = ChannelGate(in_channels)
            if i != n_layers - 1:
                layers += [nn.PReLU()]
            in_channels = out_channels
        return nn.Sequential(*layers)
            
    def forward(self, x: torch.Tensor):
        if not self.first:
            x = self.pool(x)
        x = self.layers(x)
        x = self.gate(x)
        return self.prelu(x)
        
    
class CNN(nn.Module):
    
    def __init__(self, pretrain, init_weights=False, num_scales=1):
        super(CNN, self).__init__()

        self.pretrain = pretrain
        # architecture
        self.encoder = Encoder([1, 64, 128, 192, 256], [64, 128, 192, 256, 256], [2, 2, 3, 3, 3], [None, 5, 5, 5, 5])
            
        self.fp_dim = 128
        self.num_scales = num_scales
        self.conv_c5 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)

        if self.num_scales > 1:
            self.conv_c4 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)
        
        if self.num_scales > 2:
            self.conv_c3 = nn.Conv1d(192, self.fp_dim, 1, 1, 0)
        
        if init_weights:
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

        c3, c4, c5 = self.encoder(x)

        if self.pretrain:
            out.append(c5)
        else:
            p5 = self.conv_c5(c5)
            out.append(p5)
            if self.num_scales > 1:
                p4 = self.conv_c4(c4)
                out.append(p4)
            if self.num_scales > 2:
                p3 = self.conv_c3(c3)
                out.append(p3)
        
        return out #(pretrain ? 1 : num_scales, c5.shape)

class MaxPool1d(nn.Module):
    def __init__(self, maxpool_size):
        super(MaxPool1d, self).__init__()
        self.maxpool_size = maxpool_size
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)

    def forward(self, x):
        _, _, n_samples = x.size()
        if n_samples % self.maxpool_size != 0:
            pad_size = self.maxpool_size - (n_samples % self.maxpool_size)
            if pad_size % 2 != 0:
                left_pad = pad_size // 2
                right_pad = pad_size // 2 + 1
            else:
                left_pad = pad_size // 2
                right_pad = pad_size // 2
            x = F.pad(x, (left_pad, right_pad), mode='constant')

        x = self.maxpool(x)

        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        print(f"In Gate channel shape {x.shape}")
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        return x * scale

if __name__ == '__main__':
    x0 = torch.randn((1, 1, 800))
    m0 = CNN(pretrain=True, init_weights=False, num_scales=1)
    forw = m0.forward(x0)
    print(forw)