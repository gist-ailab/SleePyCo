import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib
import numpy as np
import pandas as pd
from scipy import signal

from scipy.io import loadmat
import wfdb
import mat73
from scipy.signal import resample
from sleepstage import stage_dict
from logger import get_logger


# Have to manually define based on the dataset
ann2label = {
    "wake": 0,
    "nonrem1": 1,
    "nonrem2": 2,
    "nonrem3": 3,
    "rem": 4,
    "undefined": 6
}

def main():
    wake_cut = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data2/Seongju/dset/Physio2018/mat",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="./C3-A2",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="C3-M2",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = get_logger(args.log_file, level="info")

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*.mat"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*arousal.mat"))
    hea_fnames = glob.glob(os.path.join(args.data_dir, "*.hea"))
    psg_fnames.sort()
    ann_fnames.sort()
    hea_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)
    psg_fnames = np.setdiff1d(psg_fnames, ann_fnames)

    for i in range(len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = loadmat(psg_fnames[i])['val']
        ann_f = mat73.loadmat(ann_fnames[i])['data']['sleep_stages']
        hea_f = wfdb.rdheader(hea_fnames[i].replace('.hea', '')).__dict__
        file_duration = hea_f['sig_len'] // hea_f['fs']
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = 30
        select_ch_idx = hea_f['sig_name'].index(args.select_ch)

        sampling_rate = hea_f['fs']
        target_sampling_rate = 100
        n_epoch_samples = int(epoch_duration * target_sampling_rate)

        signals = psg_f[select_ch_idx]
        ann_bin = np.zeros_like(signals)
        ann_label = np.zeros_like(signals)
        for key in ann_f.keys():
            ann_bin += ann_f[key]
            ann_label += ann_f[key] * ann2label[key]
        
        assert np.max(ann_bin) == 1
        if sampling_rate != target_sampling_rate:
            signals = resample(signals, (len(signals) // sampling_rate) * target_sampling_rate)
        
        stack = 0
        ann_stages = []
        ann_durations = []

        for j in range(len(ann_label)):
            if j == 0:
                stack += 1
            else:
                if ann_label[j - 1] == ann_label[j]:
                    stack += 1
                else:
                    stack += 1
                    ann_stages.append(ann_label[j - 1])
                    ann_durations.append(stack / sampling_rate)
                    stack = 0
        
        for k in range(len(ann_durations)):
            if ann_durations[k] % epoch_duration != 0:
                if k == len(ann_durations) - 1:
                    ann_durations[k] = (ann_durations[k] // epoch_duration) * epoch_duration
                else:
                    raise ValueError
        
        cut_samples = len(signals) - (len(signals) // n_epoch_samples) * n_epoch_samples
        if cut_samples != 0:
            signals = signals[:-cut_samples]
        signals = signals.reshape(-1, n_epoch_samples)

        logger.info("Select channel: {}".format(select_ch))
        logger.info("Select channel samples: {}".format(signals))
        logger.info("Sample rate: {}".format(sampling_rate))

        labels = []
        onset_sec = 0.
        
        for a in range(len(ann_stages)):
            duration_sec = int(ann_durations[a])

            # Get label value
            label = ann_stages[a]
            ann_str = list(ann2label.keys())[list(ann2label.values()).index(label)]

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                #raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)

            # logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
            #     onset_sec, duration_sec, label, ann_str
            # ))
            onset_sec += duration_sec + ann_durations[a]
        labels = np.hstack(labels)
        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals)]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)
        
        if wake_cut:
            # Select only sleep periods
            w_edge_mins = 30
            nw_idx = np.where(y != 0)[0]
            start_idx = nw_idx[0] - (w_edge_mins * 2)
            end_idx = nw_idx[-1] + (w_edge_mins * 2)
            if start_idx < 0: start_idx = 0
            if end_idx >= len(y): end_idx = len(y) - 1
            select_idx = np.arange(start_idx, end_idx+1)
            logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
            x = x[select_idx]
            y = y[select_idx]
            logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

        # Remove movement and unknown
        move_idx = np.where(y == 5)[0]
        unk_idx = np.where(y == 6)[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            logger.info("Remove irrelavant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]
            logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x, 
            "y": y, 
            "fs": sampling_rate,
            "ch_label": select_ch,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_epochs": len(x)
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        logger.info("\n=======================================\n")


if __name__ == "__main__":
    main()

