import torch.nn as nn
import torch.nn.functional as F

from .sleepyco import SleePyCoBackbone
from .xsleepnet import XSleepNetFeature
from .iitnet import IITNetBackbone
from .utime import UTimeEncoder
from .deepsleepnet import DeepSleepNetFeature
from .tinysleepnet import TinySleepNetFeature

from .classifiers import get_classifier


last_chn_dict = {
    'SleePyCo': 256,
    'XSleepNet': 256,
    'IITNet': 128,
    'UTime': 256,
    'DeepSleepNet': 128,
    'TinySleepNet': 128
}


class MainModel(nn.Module):
    
    def __init__(self, config):

        super(MainModel, self).__init__()

        self.cfg = config
        self.bb_cfg = config['backbone']
        self.training_mode = config['training_params']['mode']

        if self.bb_cfg['name'] == 'SleePyCo':
            self.feature = SleePyCoBackbone(self.cfg)
        elif self.bb_cfg['name'] == 'XSleepNet':
            self.feature = XSleepNetFeature(self.cfg)
        elif self.bb_cfg['name'] == 'UTime':
            self.feature = UTimeEncoder(self.cfg)
        elif self.bb_cfg['name'] == 'IITNet':
            self.feature = IITNetBackbone(self.cfg)
        elif self.bb_cfg['name'] == 'DeepSleepNet':
            self.feature = DeepSleepNetFeature(self.cfg)
        elif self.bb_cfg['name'] == 'TinySleepNet':
            self.feature = TinySleepNetFeature(self.cfg)
        else:
            raise NotImplementedError('backbone not supported: {}'.format(config['backbone']['name']))

        if self.bb_cfg['dropout']:
            self.dropout = nn.Dropout(p=0.5)

        if self.training_mode == 'pretrain':
            proj_dim = self.cfg['proj_head']['dim']
            if config['proj_head']['name'] == 'Linear':
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(last_chn_dict[config['backbone']['name']], proj_dim)
                )
            elif config['proj_head']['name'] == 'MLP':
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(last_chn_dict[config['backbone']['name']], proj_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(proj_dim, proj_dim)
                )
            else:
                raise NotImplementedError('head not supported: {}'.format(config['proj_head']['name']))
            
            print('[INFO] Number of params of backbone: ', sum(p.numel() for p in self.feature.parameters() if p.requires_grad))
            print('[INFO] Number of params of proj_head: ', sum(p.numel() for p in self.head.parameters() if p.requires_grad))

        elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            self.classifier = get_classifier(config)

            print('[INFO] Number of params of backbone: ', sum(p.numel() for p in self.feature.parameters() if p.requires_grad))
            print('[INFO] Number of params of classifier: ', sum(p.numel() for p in self.classifier.parameters() if p.requires_grad))
            
        else:
            raise NotImplementedError('head not supported: {}'.format(config['training_params']['mode']))           

    def get_max_len(self, features):
        len_list = []
        for feature in features:
            len_list.append(feature.shape[1])
        
        return max(len_list)

    def forward(self, x):
        outputs = []
        features = self.feature(x)
        
        for feature in features:
            if self.bb_cfg['dropout']:
                feature = self.dropout(feature)
                
            if self.training_mode == 'pretrain':
                outputs.append(F.normalize(self.head(feature)))
            elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
                feature = feature.transpose(1, 2)
                output = self.classifier(feature)
                outputs.append(output)    # (B, L, H)
            else:
                raise NotImplementedError
            
        return outputs
