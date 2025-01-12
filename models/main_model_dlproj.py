import torch.nn as nn

from models.cnn_attention.cnn_attention import CnnBackboneWithAttn
from models.transformer.transformerbackbone import TransformerBackbone

from models.classifiers import get_classifier
from models.cnn.cnn_backbone import CnnBackbone


class MainModelDLProject(nn.Module):
    """
    This is the Main Model for working with Backbone Encoders trained via Masked Prediction Tasks.
    It supports 2 modi:

    1. pretrain_mp: pretraining of a backbone encoder via masked prediction via a masked Autoencoder
    2. pretrain
    3. train-classifier
    4. classification:
    4. pretrain-hybrid:

    In mode 1, the Model loaded needs to have the following outputs: reconstructed epoch (so full autoencoder must work)
    In mode 2, the Model loaded needs to have the following outputs: latent space representation (so the decoder is not needed)
    -> This needs to be configurable in every model!
    """

    def __init__(self, config):

        super(MainModelDLProject, self).__init__()

        self.cfg = config
        self.bb_cfg = config['backbone']
        self.training_mode = config['training_params']['mode']

        if self.bb_cfg['name'] == 'Transformer':
            self.backbone = TransformerBackbone(self.training_mode, self.bb_cfg)
        elif self.bb_cfg['name'] == 'CnnOnly':
            self.backbone = CnnBackbone(self.training_mode, self.bb_cfg)
        elif self.bb_cfg['name'] == 'Cnn+Attention':
            self.backbone = CnnBackboneWithAttn(self.training_mode, self.bb_cfg)
        else:
            raise NotImplementedError('backbone not supported: {}'.format(config['backbone']['name']))
        self.classifier = None
        # make sure operating mode is supported by model
        self.switch_mode(self.training_mode)


    def activate_train_mode(self):
        self.backbone.train()
        if self.classifier is not None:
            self.classifier.train()

        if self.training_mode == 'train-classifier':
            self.freeze_backbone()


    def switch_mode(self, mode):
        self.training_mode = mode
        assert self.training_mode in ["pretrain-hybrid", "pretrain_mp", "pretrain", "train-classifier", "classification", "gen-embeddings"]
        assert self.backbone.is_mode_supported(self.training_mode)

        if self.training_mode == "gen-embeddings":
            self.freeze_backbone()

        if self.training_mode in ['train-classifier', 'classification']:
            # allowed other modi attach a classifier on only the encoder to perform benchmarks, train classifier or finetune
            self.classifier = get_classifier(self.cfg)
            self.freeze_backbone()
            if self.training_mode == 'classification':
                self.freeze_classifier()
            print('[INFO] #Parameters of classifier (Only those requiring gradient): ',
                  sum(p.numel() for p in self.classifier.parameters() if p.requires_grad))
        # change backbone model mode
        self.backbone.switch_mode(self.training_mode)
        print('[INFO] #Parameters of backbone (Only those requiring gradient): ',
              sum(p.numel() for p in self.backbone.parameters() if p.requires_grad))


    def freeze_classifier(self):
        print("[WARNING] Freezing classifier weights...")
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier.eval()

    def freeze_backbone(self):
        print("[WARNING] Freezing backbone weights and setting backbone in eval mode...")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, x):
        """
        Directly forwards the output of the model (outputs or loss). Loss in case where model is calculating loss internally
        and we are in training mode. Output in all other cases.
        """
        output = self.backbone(x)  # depending on what mode is used, it can be latent or reconstruction or feature pyramid

        if self.training_mode in ['train-classifier', 'classification']:
            # latent output gets used to classify
            output = self.classifier(output)

        # wrap output in list if not already done in model itself
        if not type(output) is list:
            output = [output]
        return output
