import argparse
import glob
import math
import ntpath
import os
import re
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

rnk2aasm = {
    0 : 0,
    1 : 1,
    2 : 2,
    3 : 3,
    4 : 3,
    5 : 4,
    6 : 5,
    9 : 9
}

label2ann = {
    0 : "Sleep stage W",
    1 : "Sleep stage 1",
    2 : "Sleep stage 2",
    3 : "Sleep stage 3",
    4 : "Sleep stage R",
    6 : "Sleep stage ?",
    5 : "Movement time",
    9 : "Error"
}

def read_annot_regex(filename):
    with open(filename, 'r') as f:
        content = f.read()
    # Check that there is only one 'Start time' and that it is 0
    patterns_start = re.findall(
        r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>', 
        content)
    assert len(patterns_start) == 1
    # Now decode file: find all occurences of EventTypes marking sleep stage annotations
    patterns_stages = re.findall(
        r'<EventType>Stages.Stages</EventType>\n' +
        r'<EventConcept>.+</EventConcept>\n' +
        r'<Start>[0-9\.]+</Start>\n' +
        r'<Duration>[0-9\.]+</Duration>', 
        content)
    # print(patterns_stages[-1])
    stages = []
    starts = []
    durations = []
    for pattern in patterns_stages:
        lines = pattern.splitlines()
        stageline = lines[1]
        stage = int(stageline[-16])
        startline = lines[2]
        start = float(startline[7:-8])
        durationline = lines[3]
        duration = float(durationline[10:-11])
        assert duration % 30 == 0.
        epochs_duration = int(duration) // 30

        stages += [stage]*epochs_duration
        starts += [start]
        durations += [duration]
    # last 'start' and 'duration' are still in mem
    # verify that we have not missed stuff..
    assert int((start + duration)/30) == len(stages)
    return stages


def main():
    wake_cut = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data2/Seongju/dset/SHHS/edf",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="./C4-A1",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG",
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
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir.replace('edf', 'xml'), "*xml"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    assert len(psg_fnames) == len(ann_fnames)

    for i in range(4600, len(psg_fnames)):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))
        psg_f = pyedflib.EdfReader(psg_fnames[i])
        ann_f = read_annot_regex(ann_fnames[i])

        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = 30.0

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels()
        ch_samples = psg_f.getNSamples()

        select_ch_idx = -1
        for s in range(psg_f.signals_in_file):
            print(ch_names[s])
            # if ch_names[s] == select_ch:
            #     select_ch_idx = s
            #     break
        exit()
        if select_ch_idx == -1:
            raise Exception("Channel not found.")
        sampling_rate = psg_f.getSampleFrequency(select_ch_idx)
        signals = psg_f.readSignal(select_ch_idx)

        assert len(signals) == len(ann_f) * sampling_rate * 30

        target_sampling_rate = 100
        n_epoch_samples = int(epoch_duration * target_sampling_rate)

        if sampling_rate != target_sampling_rate:
            signals = resample(signals, int((len(signals) // sampling_rate) * target_sampling_rate))

        logger.info("Select channel: {}".format(select_ch))
        logger.info("Select channel samples: {}".format(signals))
        logger.info("Sample rate: {}".format(target_sampling_rate))

        signals = signals.reshape(-1, n_epoch_samples)

        # Generate labels from onset and duration annotation
        labels = []
        onset_sec = 0.
        for a in range(len(ann_f)):
            label = rnk2aasm[ann_f[a]]
            duration_epoch = int(30.0 / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)
            ann_str = label2ann[label]

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, epoch_duration, label, ann_str
            ))
        labels = np.hstack(labels)

        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals)]

        # Get epochs and their corresponding labels
        x = signals.astype(np.float32)
        y = labels.astype(np.int32)

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
        err_idx = np.where(y == 9)[0]
        if len(move_idx) > 0 or len(unk_idx) > 0 or len(err_idx) > 0:
            remove_idx = np.union1d(np.union1d(move_idx, unk_idx), err_idx)
            logger.info("Remove irrelavant stages")
            logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
            logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
            logger.info("  Error: ({}) {}".format(len(err_idx), err_idx))
            logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
            logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
            select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
            x = x[select_idx]
            y = y[select_idx]
            logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace(".edf", ".npz")
        save_dict = {
            "x": x, 
            "y": y, 
            "fs": sampling_rate,
            "ch_label": select_ch,
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_epochs": len(x),
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        logger.info("\n=======================================\n")


if __name__ == "__main__":
    main()
