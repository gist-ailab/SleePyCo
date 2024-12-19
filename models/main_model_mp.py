import torch.nn as nn
import torch.nn.functional as F

from models.transformer.transformerbackbone import TransformerBackbone

from .classifiers import get_classifier


last_chn_dict = {
    'SleePyCo': 256,
    'XSleepNet': 256,
    'IITNet': 128,
    'UTime': 256,
    'DeepSleepNet': 128,
    'TinySleepNet': 128
}

#  TODO: Invent new mode for latent space benchmarks via output of latent representation?
class MainModelMaskedPrediction(nn.Module):
    """
    This is the Main Model for working with Backbone Encoders trained via Masked Prediction Tasks.
    It supports 2 modi:

    1. pretrain_mp: pretraining of a backbone encoder via masked prediction via a masked Autoencoder
    2. 'scratch', 'fullfinetune', 'freezefinetune': perform benchmarks or classification or finetuning

    In mode 1, the Model loaded needs to have the following outputs: reconstructed epoch (so full autoencoder must work)
    In mode 2, the Model loaded needs to have the following outputs: latent space representation (so the decoder is not needed)
    -> This needs to be configurable in every model!
    """

    def __init__(self, config):

        super(MainModelMaskedPrediction, self).__init__()

        self.cfg = config
        self.bb_cfg = config['backbone']
        self.training_mode = config['training_params']['mode']

        if self.bb_cfg['name'] == 'Transformer':
            self.model = TransformerBackbone(self.mode, self.bb_cfg)
        elif self.bb_cfg['name'] == 'CnnOnly':
            self.model = None
        else:
            raise NotImplementedError('backbone not supported: {}'.format(config['backbone']['name']))
        # make sure operating mode is supported by model
        assert self.model.is_mode_supported(self.training_mode)
        print('[INFO] Number of params of backbone: ',
              sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        if self.bb_cfg['dropout']:
            self.dropout = nn.Dropout(p=0.5)


        if self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            # allowed other modi attach a classifier on only the encoder to perform benchmarks, train classifier or finetune
            self.classifier = get_classifier(config)
            print('[INFO] Number of params of classifier: ', sum(p.numel() for p in self.classifier.parameters() if p.requires_grad))
            
        else:
            raise NotImplementedError('Train Mode not supported: {}'.format(config['training_params']['mode']))

    def get_max_len(self, features):
        len_list = []
        for feature in features:
            len_list.append(feature.shape[1])
        
        return max(len_list)

    def forward(self, x):
        outputs = []
        output = self.model(x)  # depending on what mode is used, it can be latent or reconstruction

        if self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            # latent output gets used to classify
            outputs = self.classifier(output)

        return outputs
