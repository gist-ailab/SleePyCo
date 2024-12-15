import torch.nn as nn
import torch.nn.functional as F

from .cnn_backbone import CNNBackbone

class CNN(nn.Module):
    
    def __init__(self, dropout=False, pretrain=True):

        super(CNN, self).__init__()

        self.feature = CNNBackbone(pretrain, init_weights=False)

        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        proj_dim = 128
        if self.pretrain:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim)
                )
            
            print('[INFO] Number of params of backbone: ', sum(p.numel() for p in self.feature.parameters() if p.requires_grad))
            print('[INFO] Number of params of proj_head: ', sum(p.numel() for p in self.head.parameters() if p.requires_grad))


    def get_max_len(self, features):
        len_list = []
        for feature in features:
            len_list.append(feature.shape[1])
        
        return max(len_list)

    def forward(self, x):
        outputs = []
        features = self.feature(x)
        
        for feature in features:
            if self.dropout:
                feature = self.dropout(feature)
                
            if self.pretrain:
                outputs.append(F.normalize(self.head(feature)))
            
        return outputs  # (B, L, H)