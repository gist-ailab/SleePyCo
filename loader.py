import os
import glob

from data_utils import SUPPORTED_MASKING, MASKING_MAP
from transform import *
from torch.utils.data import Dataset


class EEGDataLoader(Dataset):
    """
    Dataloader to load single channel EEG signals. It can operate in different training modi:
        1. pretrain: from one eeg epoch, generate two views using augmentations
        2. others: ('scratch', 'fullyfinetune', 'freezefinetune') - only load single channel EEG signals, no two views
        3. pretrain_mp: use this one for semantic clearness when doing pretraining with masked prediction. Here we only use single channel EEG signals as in "other" TODO: potentially adding masking here
    """

    def __init__(self, config, fold, set='train'):

        self.set = set  # train or val
        self.fold = fold  # idx of fold from cross-validation (influences how split)

        self.sampling_rate = 100 # how many values sampled per second from EEG recording
        self.dset_cfg = config['dataset']
        
        self.root_dir = self.dset_cfg['root_dir']
        self.dset_name = self.dset_cfg['name']
        self.num_splits = self.dset_cfg['num_splits']
        self.eeg_channel = self.dset_cfg['eeg_channel']
        
        self.seq_len = self.dset_cfg['seq_len']
        self.target_idx = self.dset_cfg['target_idx']
        
        self.training_mode = config['training_params']['mode']

        self.dataset_path = os.path.join(self.root_dir, 'dset', self.dset_name, 'npz')
        self.inputs, self.labels, self.epochs = self.split_dataset()

        # Setup masking if activated
        self.masking_activated = "masking" in self.dset_cfg.keys() and self.dset_cfg["masking"]
        if self.masking_activated:
            # check for set masking mode and store respective method
            assert "masking_type" in self.dset_cfg.keys()
            self.masking_type = self.dset_cfg["masking_type"]
            assert self.masking_type in SUPPORTED_MASKING
            self.masking_method = MASKING_MAP[self.masking_type]
            assert "masking_ratio" in self.dset_cfg.keys()
            self.masking_ratio = self.dset_cfg["masking_ratio"]

        # Setup two-transform in case of contrastive learning
        if self.training_mode == 'pretrain':
            self.transform = Compose(
                transforms=[
                    RandomAmplitudeScale(),
                    RandomTimeShift(),
                    RandomDCShift(),
                    RandomZeroMasking(),
                    RandomAdditiveGaussianNoise(),
                    RandomBandStopFilter(),
                ]
            )
            self.two_transform = TwoTransform(self.transform)
        
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        n_sample = 30 * self.sampling_rate * self.seq_len
        file_idx, idx, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][idx:idx+seq_len]

        if self.set == 'train':
            if self.training_mode == 'pretrain':
                assert seq_len == 1
                input_a, input_b = self.two_transform(inputs)
                input_a = torch.from_numpy(input_a).float()
                input_b = torch.from_numpy(input_b).float()
                inputs = [input_a, input_b]
            else:
                inputs = inputs.reshape(1, n_sample)
                inputs = torch.from_numpy(inputs).float()
        else:
            if not self.training_mode == 'pretrain':
                inputs = inputs.reshape(1, n_sample)
            inputs = torch.from_numpy(inputs).float()
        
        labels = self.labels[file_idx][idx:idx+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]

        if self.masking_activated:
            inputs, masked_inputs, mask = self.masking_method(inputs, self.masking_ratio)
            return {'inputs': inputs, 'masked_inputs': masked_inputs, 'mask': mask}, labels

        return inputs, labels

    def split_dataset(self):

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dset_name)), allow_pickle=True)

        assert len(split_idx_list) == self.num_splits
    
        if self.dset_name == 'Sleep-EDF-2013':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx == self.fold - 1:
                    data_fname_dict['test'].append(data_fname_list[i])
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict['val'].append(data_fname_list[i])
                else:
                    data_fname_dict['train'].append(data_fname_list[i])    

        elif self.dset_name == 'Sleep-EDF-2018':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
                    
        elif self.dset_name == 'MASS' or self.dset_name == 'Physio2018' or self.dset_name == 'SHHS':
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))
            
        for data_fname in data_fname_dict[self.set]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            seq_len = self.seq_len
            if self.dset_name== 'MASS' and ('-02-' in data_fname or '-04-' in data_fname or '-05-' in data_fname):
                seq_len = int(self.seq_len * 1.5)
            for i in range(len(npz_file['y']) - seq_len + 1):
                epochs.append([file_idx, i, seq_len])
            file_idx += 1
        
        return inputs, labels, epochs
