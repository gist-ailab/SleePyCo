import torch.nn as nn


class XSleepNetFeature(nn.Module):
    def __init__(self, config):
        super(XSleepNetFeature, self).__init__()

        self.training_mode = config['training_params']['mode']
        
        # architecture
        self.conv1 = self.make_layers(1, 16)
        self.conv2 = self.make_layers(16, 16)
        self.conv3 = self.make_layers(16, 32)
        self.conv4 = self.make_layers(32, 32)
        self.conv5 = self.make_layers(32, 64)
        self.conv6 = self.make_layers(64, 64)
        self.conv7 = self.make_layers(64, 128)
        self.conv8 = self.make_layers(128, 128)
        self.conv9 = self.make_layers(128, 256)

        if self.training_mode == 'freezefinetune' or self.training_mode == 'scratch':
            self.fp_dim = config['feature_pyramid']['dim']
            self.num_scales = config['feature_pyramid']['num_scales']
            self.conv_c5 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)

            if self.num_scales > 1:
                self.conv_c4 = nn.Conv1d(128, self.fp_dim, 1, 1, 0)
            
            if self.num_scales > 2:
                self.conv_c3 = nn.Conv1d(128, self.fp_dim, 1, 1, 0)
            
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

    def make_layers(self, in_channels, out_channels):
        layer = [nn.Conv1d(in_channels, out_channels, 31, 2, 15)]
        layer.append(nn.BatchNorm1d(out_channels))
        layer.append(nn.PReLU())

        return nn.Sequential(*layer)

    def forward(self, x):
        out = []

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        c1 = self.conv5(x)
        c2 = self.conv6(c1)
        c3 = self.conv7(c2)
        c4 = self.conv8(c3)
        c5 = self.conv9(c4)

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
