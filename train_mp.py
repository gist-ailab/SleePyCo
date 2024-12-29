"""
This file contains training logic for the Masked Prediction Paradigm. Aims to allow training as in MAEEG or EEg2Rep.
Usage:
    - Create a config that specifies dataset, model and loss function
    - dataset: make sure to use the correct training mode so dataloader and model behave correctly
        - ex. pretrain_mp lets dataloader only load single eeg epochs not two-views augmented!
    - loss: specify a correct loss function. Add a supported loss name as 'loss' to train config
        - you can additionally also add loss-params a dict of named arguments you pass to your loss
"""

import json
import argparse
import warnings

import torch.optim as optim
from torch.utils.data import DataLoader

from models.main_model_mp import MainModelMaskedPrediction
from utils import *
from loss import *
from loader import EEGDataLoader


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']

        # assert that the correct training mode is set: 'pretrain_mp'. THis makes sure the models and dataset show correct behavior
        assert self.tp_cfg['mode'] and self.tp_cfg['mode'] == 'pretrain_mp'
        # assert a loss is given in the current training config and the loss exists
        assert self.tp_cfg.get('loss', False)
        assert self.tp_cfg['loss'] in SUPPORTED_LOSS_FUNCTIONS

        self.device = get_device(preference="cuda")
        print('[INFO] Config name: {}'.format(config['name']))
        print('[INFO] Device: {}'.format(str(self.device)))
        
        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        # load selected loss with its parameters (if these parameters are given)
        self.criterion = LOSS_MAP[self.tp_cfg['loss']](**(self.tp_cfg['loss_params'] if 'loss_params' in self.tp_cfg.keys() else {}))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

    def build_model(self):
        model = MainModelMaskedPrediction(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        dataloader_args = {'batch_size': self.tp_cfg['batch_size'], # default data loader args, using 4 workers per GPU, TODO: what about prefetching
                           'shuffle': True,
                           'num_workers': 4*len(self.args.gpu.split(",")), # self.args.gpu.split defaults to 1 even when arg not given
                           'pin_memory': True}
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader}

    def train_one_epoch(self):
        """
        We modify this method from train_crl.py to be able to deal with only raw single eeg epochs and not the two-view
        augmented approach.
        """
        self.model.train()
        train_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            # input-shape:(B, 1, 3000), labels.shape: (B,)
            loss = 0
            labels = labels.view(-1).to(self.device) # TODO: No effect here
            #inputs = inputs.squeeze().to(self.device) # squeeze middle dimension
            inputs = inputs.to(self.device) # TODO: CNN needs middle dim ex. (B, 1, 3000), but transformer dont! is this intentional? + what is this middle dim?

            # outputs here are expected to be reconstruction prediction. output is list of one element in pretrain_mp mode
            outputs = self.model(inputs)[0]

            # outputs shape: (B, 51, 1472)
            loss += self.criterion(inputs, outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Lr: %.4e | Loss: %.3f' %(get_lr(self.optimizer), train_loss / (i + 1)))

            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_loss = self.evaluate(mode='val')
                self.early_stopping(None, val_loss, self.model)
                self.model.train()
                if self.early_stopping.early_stop:
                    break

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        eval_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            # inputs-shape: (B, 1, 3000), labels shape: (B,)
            loss = 0
            inputs = inputs.squeeze().to(self.device)
            labels = labels.view(-1).to(self.device)
            
            outputs = self.model(inputs)[0]

            features = outputs.unsqueeze(1).repeat(1, 2, 1)
            loss += self.criterion(inputs, features, labels)

            eval_loss += loss.item()
            
            progress_bar(i, len(self.loader_dict[mode]), 'Lr: %.4e | Loss: %.3f' %(get_lr(self.optimizer), eval_loss / (i + 1)))

        return eval_loss
    
    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch()
            if self.early_stopping.early_stop:
                break

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    
    for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        trainer.run()


if __name__ == "__main__":
    main()
