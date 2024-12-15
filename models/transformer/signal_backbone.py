# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class SignalBackBone(nn.Module):
    def __init__(self, fs: int, window: int):
        super().__init__()
        self.model = BackBone(input_size=fs * window, input_channel=1, layers=[1, 1, 1, 1])
        self.feature_num = self.model.get_final_length() // 2
        self.feature_layer = nn.Sequential(
            nn.Linear(self.model.get_final_length(), self.feature_num),
            nn.ELU(),
            nn.Linear(self.feature_num, self.feature_num)
        )

    def forward(self, x):
        latent_seq = []
        for i in range(x.shape[1]):
            sample = torch.unsqueeze(x[:, i, :], dim=1)
            latent = self.model(sample)
            latent_seq.append(latent)
        latent_seq = torch.stack(latent_seq, dim=1)
        latent_seq = self.feature_layer(latent_seq)
        return latent_seq


class BackBone(nn.Module):
    def __init__(self, input_size, input_channel, layers):
        super().__init__()
        self.inplanes3 = 32
        self.inplanes5 = 32
        self.inplanes7 = 32

        self.input_size = input_size
        self.conv1 = nn.Conv1d(input_channel, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 32, layers[0], stride=1)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 32, layers[1], stride=1)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 48, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 64, layers[3], stride=2)
        self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)

        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 32, layers[0], stride=1)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 32, layers[1], stride=1)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 48, layers[2], stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 64, layers[3], stride=2)
        self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 32, layers[0], stride=1)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 32, layers[1], stride=1)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 48, layers[2], stride=2)
        self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 64, layers[3], stride=2)
        self.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)

    def forward(self, x0):
        b = x0.shape[0]
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer3x3_1(x0)
        x1 = self.layer3x3_2(x1)
        x1 = self.layer3x3_3(x1)
        x1 = self.layer3x3_4(x1)
        x1 = self.maxpool3(x1)

        x2 = self.layer5x5_1(x0)
        x2 = self.layer5x5_2(x2)
        x2 = self.layer5x5_3(x2)
        x2 = self.layer5x5_4(x2)
        x2 = self.maxpool5(x2)

        x3 = self.layer7x7_1(x0)
        x3 = self.layer7x7_2(x3)
        x3 = self.layer7x7_3(x3)
        x3 = self.layer7x7_4(x3)
        x3 = self.maxpool7(x3)

        out = torch.cat([x1, x2, x3], dim=-1)
        out = torch.reshape(out, [b, -1])
        return out

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def get_final_length(self):
        x = torch.randn(1, 1, self.input_size)
        x = self.forward(x)
        return x.shape[-1]


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        return out1


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


if __name__ == '__main__':
    # st = ST_BackBone(input_size=500)
    # ss = st(
    #     torch.randn(50, 1, 500)
    # )
    fb = SignalBackBone(fs=100, window=3)
    ss = fb(
        torch.randn(50, 1, 300)
    )
    print(ss.shape)

