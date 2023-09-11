from torch.utils.data import DataLoader
import csv
from torch.utils.data.sampler import *

from .data_util import get_filelists_from_csv
from .real_CR_dataset import Real_CR_Dataset
from .test_CR_dataset import Test_CR_Dataset
# from .sen12_CR_dataset import SEN12_CR_Dataset


def getLoader(configure_dataset):
    opt = configure_dataset

    if opt['name'] == 'Sen12_CR':
        pass
        # dataset = SEN12_CR_Dataset(opt)
    elif opt['name'] == 'Real_CR':
        dataset = Real_CR_Dataset(opt)
    elif opt['name'] == 'Test_CR':
        dataset = Test_CR_Dataset(opt)
    else:
        raise NotImplementedError(f"Dataset {opt['name']} not implement!")

    bs = opt['batch_size']
    use_shuffle = opt['use_shuffle'] if 'use_shuffle' in opt else False
    use_drop_last = opt['use_drop_last'] if 'use_drop_last' in opt else False
    weighted_sampler = opt['weighted_sampler'] if 'weighted_sampler' in opt else None

    if weighted_sampler:
        num_samples = int(len(dataset) * weighted_sampler['sampler_ratio'])
        sampler = WeightedRandomSampler(dataset.sampler_weights, num_samples=num_samples, replacement=False)
        return DataLoader(dataset, batch_size=bs, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=bs, shuffle=use_shuffle, drop_last=use_drop_last)


def getLoaders(configure_datasets):
    loaders = []
    for dataset in configure_datasets:
        loaders.append(getLoader(configure_datasets[dataset]))

    return loaders[0], loaders[1:]

def get_filelists_from_csv(listpath, phase):
    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)

    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)
    csv_file.close()
    # to do
    if phase == 'train':
        return train_filelist
    elif phase == 'val':
        return val_filelist
    elif phase == 'test':
        return test_filelist