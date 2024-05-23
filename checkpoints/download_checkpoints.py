import os
import gdown
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datasets', nargs='+', type=str, default='all')
args = parser.parse_args()
print(args.datasets)

# Sleep-EDF-2013
if args.datasets == 'all' or 'Sleep-EDF-2013' in args.datasets:
    file_id = '1oUs8S9dVwmTJi9t9zh7msmJT_B28OpbP'
    config_name = 'SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_freezefinetune'
    gdown.download(id=file_id, output=config_name + '.zip', quiet=False)
    os.system('unzip {}.zip -d {}'.format(config_name, config_name))

# Sleep-EDF-2018
if args.datasets == 'all' or 'Sleep-EDF-2018' in args.datasets:
    file_id = '1RdWl9AUMkFlNwUE2qxx3v5XcL3Exs0Pk'
    config_name = 'SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune'
    gdown.download(id=file_id, output=config_name + '.zip', quiet=False)
    os.system('unzip {}.zip -d {}'.format(config_name, config_name))

# MASS
if args.datasets == 'all' or 'MASS' in args.datasets:
    file_id = '16kPPhW04g5swGQeOJs8aRJOI13wSEKhI'
    config_name = 'SleePyCo-Transformer_SL-10_numScales-3_MASS_freezefinetune'
    gdown.download(id=file_id, output=config_name + '.zip', quiet=False)
    os.system('unzip {}.zip -d {}'.format(config_name, config_name))

# Physio2018
if args.datasets == 'all' or 'Physio2018' in args.datasets:
    file_id = '1r4NXeSzmP5rp_WTTGxiwHLGzknjPV8PT'
    config_name = 'SleePyCo-Transformer_SL-10_numScales-3_Physio2018_freezefinetune'
    gdown.download(id=file_id, output=config_name + '.zip', quiet=False)
    os.system('unzip {}.zip -d {}'.format(config_name, config_name))

# SHHS
if args.datasets == 'all' or 'SHHS' in args.datasets:
    file_id = '1FwjtO3JLd1Di0yRmz7g4B0niyY0gzQEd'
    config_name = 'SleePyCo-Transformer_SL-10_numScales-3_SHHS-2018_freezefinetune'
    gdown.download(id=file_id, output=config_name + '.zip', quiet=False)
    os.system('unzip {}.zip -d {}'.format(config_name, config_name))
