import torch
import torch.nn as nn
from .utils import Conv1d, MaxPool1d


class TinySleepNetFeature(nn.Module):
    def __init__(self, config, dropout=0.5):
        super(TinySleepNetFeature, self).__init__()

        self.chn = 64
        self.training_mode = config['training_params']['mode']

        # architecture
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.path = nn.Sequential(
            Conv1d(1, self.chn*2, 50, 25, padding='SAME', bias=False),
            nn.BatchNorm1d(self.chn*2),
            nn.ReLU(),
            MaxPool1d(8, padding='SAME'),
            nn.Dropout(),
            Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
            nn.BatchNorm1d(self.chn*2),
            nn.ReLU(),
            Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
            nn.BatchNorm1d(self.chn * 2),
            nn.ReLU(),
            Conv1d(self.chn*2, self.chn*2, 8, 1, padding='SAME', bias=False),
            nn.BatchNorm1d(self.chn*2),
            nn.ReLU(),
            MaxPool1d(4, padding='SAME'),
            nn.Dropout()
            )

        if self.training_mode == 'freezefinetune' or self.training_mode == 'scratch':
            self.fp_dim = config['feature_pyramid']['dim']
            self.num_scales = config['feature_pyramid']['num_scales']
            self.conv_c5 = nn.Conv1d(128, self.fp_dim, 1, 1, 0)
        
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
        c5 = self.path(x)
        
        if self.training_mode == 'pretrain':
            out.append(c5)
        elif self.training_mode in ['scratch', 'fullyfinetune', 'freezefinetune']:
            p5 = self.conv_c5(c5)
            out.append(p5)

        return out