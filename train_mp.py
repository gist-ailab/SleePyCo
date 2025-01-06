"""
This file contains training logic for the Masked Prediction Paradigm. Aims to allow training as in MAEEG or EEg2Rep.
Usage:
    - Create a config that specifies dataset, model and loss function
    - dataset: make sure to use the correct training mode so dataloader and model behave correctly
        - ex. pretrain_mp lets dataloader only load single eeg epochs not two-views augmented!
    - loss: specify a correct loss function. Add a supported loss name as 'loss' to train config
        - you can additionally also add loss-params a dict of named arguments you pass to your loss
"""

import time
import json
import argparse
import warnings

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
        self.dset_cfg = config['dataset']

        # assert that the correct training mode is set: 'pretrain_mp'. This makes sure the models and dataset show correct behavior
        if not 'mode' in self.tp_cfg.keys() and self.tp_cfg['mode'] == 'pretrain_mp':
            raise ValueError(
                'Running train_mp.py, only mode pretrain_mp is supported and must be declared in the config file')

        self.device = get_device(preference="cuda")
        print('[INFO] Config name: {}'.format(config['name']))
        print('[INFO] Device: {}'.format(str(self.device)))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        self.backbone_ref = self.model.module.model

        # create tensorboard writer
        self.writer = SummaryWriter(log_dir=os.path.join("logs", config['name'], f"fold-{fold}"))

        # load selected loss with its parameters (if these parameters are given) - only used if model has no internal loss calculation
        if not self.backbone_ref.is_using_internal_loss():  # access backbone
            assert self.tp_cfg.get('loss', False)
            assert self.tp_cfg['loss'] in SUPPORTED_LOSS_FUNCTIONS
            self.criterion = LOSS_MAP[self.tp_cfg['loss']](
                **(self.tp_cfg['loss_params'] if 'loss_params' in self.tp_cfg.keys() else {}))
        else:
            # if internal loss is used print info
            print("Backbone loaded uses Internal loss calculation! Model itself will check for correct loss params if needed for itself...")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.tp_cfg['lr'],
                                    weight_decay=self.tp_cfg['weight_decay'])

        # check if masking is performed internally, in that case throw a warning if masking is also activated in dataset
        self.dset_masking_activated = ("masking" in self.dset_cfg.keys() and self.dset_cfg["masking"])
        if self.backbone_ref.is_using_internal_masking() and self.dset_masking_activated:
            warnings.warn("USING MASKING ON RAW DATA LEVEL AND INSIDE MODEL! IS THIS INTENTIONAL?")
            time.sleep(2)

        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path,
                                            ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

    def build_model(self):
        model = MainModelMaskedPrediction(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model

    def build_dataloader(self):
        dataloader_args = {'batch_size': self.tp_cfg['batch_size'],
                           # default data loader args, using 4 workers per GPU, also have 2 batches prefetched by default
                           'shuffle': True,
                           'num_workers': 4 * len(self.args.gpu.split(",")),
                           # self.args.gpu.split defaults to 1 even when arg not given
                           'pin_memory': True}
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        print('[INFO] Dataloader prepared')
        print('[INFO] Batch-Size: {}'.format(self.tp_cfg['batch_size']))
        print('[INFO] Train-Batches: {}, Val-Batches: {}'.format(len(train_loader), len(val_loader)))

        return {'train': train_loader, 'val': val_loader}

    def train_one_epoch(self, epoch):
        """
        We modify this method from train_crl.py to be able to deal with only raw single eeg epochs and not the two-view
        augmented approach.
        """
        self.model.train()
        train_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            # input-shape:(B, 1, 3000), labels.shape: (B,) -> dummy dim is removed in models that don't need it.

            loss = 0
            labels = labels.view(-1).to(self.device)  # No effect in our case!

            # if dset level masking is activated unpack values accordingly, currently only supported if no internal loss is calculated
            # if we want ot change that we need to change this script here!
            mask = None
            if self.dset_masking_activated:
                assert not self.backbone_ref.is_using_internal_loss()
                masked_input = inputs["masked_inputs"].to(self.device)
                mask = inputs["mask"]
                inputs = inputs["inputs"]

            inputs = inputs.to(self.device)

            if not self.backbone_ref.is_using_internal_loss():
                # Model is not using internal loss
                if self.dset_masking_activated:
                    outputs = self.model(masked_input)[0]
                else:
                    outputs = self.model(inputs)[0]

                # calculate loss based on predictions, gt and whether a mask is given or not
                loss += self.criterion(inputs, outputs, reduction='mean', mask=mask, labels=labels)
            else:
                # Model is using internal loss, so output will be loss (needed inc ase of latent masked pred or framewise loss in transformer for example)
                loss += self.model(inputs)[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar("train/loss", loss.item(), self.train_iter)
            train_loss += loss.item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']),
                         'Lr: %.4e | Loss: %.6f' % (get_lr(self.optimizer), train_loss / (i + 1)))

            # perform validation every X iterations
            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                print(f'[INFO] Starting evaluation...')
                val_loss = self.evaluate(mode='val')
                self.early_stopping(None, val_loss, self.model)
                self.model.train()
                if self.early_stopping.early_stop:
                    print("[INFO] Early stopping...")
                    break

        # Log average training loss of an epoch to TensorBoard
        avg_train_loss = train_loss / len(self.loader_dict['train'])
        self.writer.add_scalar("train/epoch-avg-loss", avg_train_loss, epoch)
        print(f"\n[INFO] Epoch {epoch}, Epochal Avg - Training Loss: {avg_train_loss:.6f}")


    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        eval_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            # input-shape:(B, 1, 3000), labels.shape: (B,) -> dummy dim is removed in models that don't need it.

            loss = 0
            labels = labels.view(-1).to(self.device)  # No effect in our case!

            # if dset level masking is activated unpack values accordingly, currently only supported if no internal loss is calculated
            # if we want ot change that we need to change this script here!
            mask = None
            if self.dset_masking_activated:
                assert not self.backbone_ref.is_using_internal_loss()
                masked_input = inputs["masked_inputs"].to(self.device)
                mask = inputs["mask"]
                inputs = inputs["inputs"]

            inputs = inputs.to(self.device)

            if not self.backbone_ref.is_using_internal_loss():
                # Model is not using internal loss
                if self.dset_masking_activated:
                    outputs = self.model(masked_input)[0]
                else:
                    outputs = self.model(inputs)[0]

                # calculate loss based on predictions, gt and whether a mask is given or not
                loss += self.criterion(inputs, outputs, reduction='mean', mask=mask, labels=labels)
            else:
                # Model is using internal loss, so output will be loss (needed inc ase of latent masked pred or framewise loss in transformer for example)
                loss += self.model(inputs)[0]

            eval_loss += loss.item()

            progress_bar(i, len(self.loader_dict[mode]),
                         'Lr: %.4e | Loss: %.4f' % (get_lr(self.optimizer), eval_loss / (i + 1)))

        avg_eval_loss = eval_loss / len(self.loader_dict[mode])
        print(f"[INFO] {mode.capitalize()} Eval-Loss: {avg_eval_loss:.4f}")
        self.writer.add_scalar(f"{mode.capitalize()}/loss-avg", avg_eval_loss, self.train_iter)
        return eval_loss

    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            if self.early_stopping.early_stop:
                break
        # close tensorboard-writer
        self.writer.close()


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

    # for our use-case we only need one split (no cross validation needed)
    trainer = OneFoldTrainer(args, 1, config)
    trainer.run()
    #for fold in range(1, config['dataset']['num_splits'] + 1):
    #    trainer = OneFoldTrainer(args, fold, config)
    #    trainer.run()


if __name__ == "__main__":
    main()
