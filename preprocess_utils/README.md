# Preprocess Utils

## Requirements

 - pyedflib
 - pandas
 - scipy
 - wfdb
 - mat73

## Usage

```bash
# for Sleep-EDF
python prepare_sleepedf.py --data_dir $SLEEP_EDF_DIR

# for MASS
python prepare_mass.py --data_dir $MASS_DIR --pretrain
python prepare_mass.py --data_dir $MASS_DIR

# for Physio2018
python prepare_physio2018.py --data_dir $PHYSIO2018_DIR

# for SHHS
python prepare_shhs.py --data_dir $SHHS_DIR
```

