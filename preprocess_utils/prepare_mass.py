import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib
import numpy as np
import pandas as pd

from scipy.signal import resample
from sleepstage import stage_dict
from logger import get_logger


# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}

def rolling_window(array, window_size,freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0,shape[0],freq)]

def main():
    total_num_epochs = 0
    wake_cut = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data2/Seongju/dset/MASS/edf/SS5",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="C4-A1",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-LER",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    parser.add_argument("--pretrain", type=bool, default=False,
                        help="Whether to use pretrain.")
    args = parser.parse_args()

    set_name = args.data_dir.split('/')[-1]

    if args.pretrain:
        args.output_dir = os.path.join(set_name, args.output_dir + '_pretrain')
    else:
        args.output_dir = os.path.join(set_name, args.output_dir + '_finetune')

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
    psg_fnames = glob.glob(os.path.join(args.data_dir, 'version2014', "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, 'annotations', "*Base.edf")) if 'SS2' in args.data_dir else glob.glob(os.path.join(args.data_dir, 'annotations', "*Annotations.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i])
        ann_f = pyedflib.EdfReader(ann_fnames[i])

        assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))
        #epoch_duration = psg_f.datarecord_duration
        if 'SS2' in args.data_dir or 'SS4' in args.data_dir or 'SS5' in args.data_dir:
            epoch_duration = 20
        else:
            epoch_duration = 30

        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            epoch_duration = epoch_duration / 2
            logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
        else:
            logger.info("Epoch duration: {} sec".format(epoch_duration))

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels()
        ch_samples = psg_f.getNSamples()
        select_ch_idx = -1
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == select_ch:
                select_ch_idx = s
                break
        if select_ch_idx == -1:
            raise Exception("Channel not found.")
        sampling_rate = psg_f.getSampleFrequency(select_ch_idx)
        target_sampling_rate = 100

        n_epoch_samples = epoch_duration * target_sampling_rate
        signals = psg_f.readSignal(select_ch_idx)

        if sampling_rate != target_sampling_rate:
            signals = resample(signals, target_sampling_rate * len(signals) // int(sampling_rate))

        logger.info("Select channel: {}".format(select_ch))
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx]))
        logger.info("Sample rate: {}".format(sampling_rate))

        # Sanity check
        n_epochs = psg_f.datarecords_in_file

        # Generate labels from onset and duration annotation
        labels = []
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
        start_idx = int(round(ann_onsets[0] * target_sampling_rate))
        end_idx = ((len(signals) - start_idx) // n_epoch_samples) * n_epoch_samples

        if pretrain and ('SS2' in args.data_dir or 'SS4' in args.data_dir or 'SS5' in args.data_dir):
            if start_idx - 5 * target_sampling_rate < 0:
                pad = np.array([0. for _ in range(5 * target_sampling_rate - start_idx)])
                signals = np.concatenate((pad, signals))
            else:
                signals = signals[start_idx - 5 * target_sampling_rate:]
            signals = signals[:end_idx + 10 * target_sampling_rate]
            if (len(signals) - 10 * target_sampling_rate) % n_epoch_samples != 0:
                pad = np.array([0. for _ in range(n_epoch_samples - ((len(signals) - 10 * target_sampling_rate) % n_epoch_samples))])
                signals = np.concatenate((signals, pad))
            signals = rolling_window(signals, 30 * target_sampling_rate, epoch_duration * target_sampling_rate)
        else:
            signals = signals[start_idx:]
            signals = signals[:end_idx]
            signals = np.reshape(signals, (-1, n_epoch_samples))

        total_duration = int(ann_onsets[0])
        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = round(ann_durations[a])
            ann_str = "".join(ann_stages[a])
            if ann_stages[a] not in ann2label.keys():
                continue
            #else:
            #    print(onset_sec)
            #    print(duration_sec)
            # Sanity check
            #assert onset_sec == total_duration

            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                #raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)

            total_duration += duration_sec

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        labels = np.hstack(labels)

        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals)]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)
       
        if len(x) != len(y):
            if len(x) > len(y):
                diff = len(x) - len(y)
                x = x[:-diff, :]
            else:
                raise NotImplementedError

        # Select only sleep periods
        if wake_cut:
            w_edge_mins = 30
            nw_idx = np.where(y != stage_dict["W"])[0]
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
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
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
        filename = ntpath.basename(psg_fnames[i])[:-4].replace(' ', '-') + '.npz'
        save_dict = {
            "x": x, 
            "y": y, 
            "fs": sampling_rate,
            "ch_label": select_ch,
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_all_epochs": n_epochs,
            "n_epochs": len(x),
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)
        total_num_epochs += len(y)
        logger.info("\n=======================================\n")
        
    logger.info('Total number of epochs: {}'.format(total_num_epochs))


if __name__ == "__main__":
    main()

