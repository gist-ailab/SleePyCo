import os
import json
import argparse
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard writer

from utils import *
from loss import SupConLoss
from loader import EEGDataLoader
from models.main_model import MainModel


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))
        
        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        self.criterion = SupConLoss(temperature=self.tp_cfg['temperature'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join("logs", config['name'], f"fold-{fold}"))
        
    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        dataloader_args = {'batch_size': self.tp_cfg['batch_size'], 'shuffle': True, 'num_workers': 4*len(self.args.gpu.split(",")), 'pin_memory': True}
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader}

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            loss = 0
            labels = labels.view(-1).to(self.device)

            inputs = torch.cat([inputs[0], inputs[1]], dim=0).to(self.device)
            outputs = self.model(inputs)[0]

            f1, f2 = torch.split(outputs, [labels.size(0), labels.size(0)], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss += self.criterion(features, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Lr: %.4e | Loss: %.3f' % (get_lr(self.optimizer), train_loss / (i + 1)))

        # Log average training loss to TensorBoard
        avg_train_loss = train_loss / len(self.loader_dict['train'])
        self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        print(f"\n[INFO] Epoch {epoch}, Training Loss: {avg_train_loss:.4f}")

        if self.train_iter % self.tp_cfg['val_period'] == 0:
            print('')
            val_loss = self.evaluate(mode='val', epoch=epoch)
            self.early_stopping(None, val_loss, self.model)
            self.model.train()
            if self.early_stopping.early_stop:
                return True
        return False

    @torch.no_grad()
    def evaluate(self, mode, epoch=None):
        self.model.eval()
        eval_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            
            outputs = self.model(inputs)[0]

            features = outputs.unsqueeze(1).repeat(1, 2, 1)
            loss += self.criterion(features, labels)

            eval_loss += loss.item()
            
            progress_bar(i, len(self.loader_dict[mode]), 'Lr: %.4e | Loss: %.3f' % (get_lr(self.optimizer), eval_loss / (i + 1)))

        avg_eval_loss = eval_loss / len(self.loader_dict[mode])
        print(f"[INFO] {mode.capitalize()} Loss: {avg_eval_loss:.4f}")

        # Log validation loss to TensorBoard
        if epoch is not None:
            self.writer.add_scalar(f"Loss/{mode.capitalize()}", avg_eval_loss, epoch)

        return avg_eval_loss
    
    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            early_stop = self.train_one_epoch(epoch)
            if early_stop:
                break

        # Close TensorBoard writer at the end
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
    
    for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        trainer.run()


if __name__ == "__main__":
    main()
