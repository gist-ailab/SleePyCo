import torch
import torch.nn as nn
from .utils import Conv1d, MaxPool1d


class DeepSleepNetFeature(nn.Module):
    def __init__(self, config):
        super(DeepSleepNetFeature, self).__init__()

        self.chn = 64
        self.training_mode = config['training_params']['mode']

        # architecture
        self.dropout = nn.Dropout(p=0.5)
        self.path1 = nn.Sequential(Conv1d(1, self.chn, 50, 6, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(8, padding='SAME'),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(4, padding='SAME')
                                   )
        self.path2 = nn.Sequential(Conv1d(1, self.chn, 400, 50, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(4, padding='SAME'),
                                   nn.Dropout(),
                                   Conv1d(self.chn, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn * 2),
                                   nn.ReLU(inplace=True),
                                   Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
                                   nn.BatchNorm1d(self.chn*2),
                                   nn.ReLU(inplace=True),
                                   MaxPool1d(2, padding='SAME'))

        self.compress = nn.Conv1d(self.chn*4, 128, 1, 1, 0)
        self.smooth = nn.Conv1d(128, 128, 3, 1, 1)

        if self.training_mode == 'freezefinetune' or self.training_mode == 'scratch':
            self.fp_dim = config['feature_pyramid']['dim']
            self.num_scales = config['feature_pyramid']['num_scales']
            self.conv_c5 = nn.Conv1d(128, self.fp_dim, 1, 1, 0)
            
            assert self.num_scales == 1

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
        x1 = self.path1(x)  # path 1
        x2 = self.path2(x)  # path 2
        
        x2 = torch.nn.functional.interpolate(x2, x1.size(2))
        c5 = self.smooth(self.compress(torch.cat([x1, x2], dim=1)))

        if self.training_mode == 'pretrain':
            out.append(c5)
        elif self.training_mode in ['scratch', 'fullyfinetune', 'freezefinetune']:
            p5 = self.conv_c5(c5)
            out.append(p5)

        return out
