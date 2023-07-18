import os
import numpy as np

import argparse
import random
import rasterio
import csv

import torch
from torch.utils.data import Dataset

from feature_detectors import get_cloud_cloudshadow_mask
from data.data_utils import get_filelists_from_csv

class SEN12_CR_Dataset(Dataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiments flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt

        self.root = opt['root']
        self.phase = opt['phase']
        self.meta_info =  opt['meta_info'] #csv

        self.random_crop = opt['random_crop'] if 'random_crop' in opt else False
        self.use_cloudmask =  opt['use_cloudmask'] if 'use_cloudmask' in opt else False

        self.base_size = opt['base_size']
        if self.random_crop:
            self.crop_size = opt['crop_size']

        if 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            self.filelist = get_filelists_from_csv(self.meta_info, self.phase)

        else:
            # todo
            pass

        self.n_images = len(self.filelist)

        self.clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.clip_max = [[0, 0], [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                    [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

        self.max_val = 1
        self.scale = 10000

    # TODO： augmentation for training
    def __getitem__(self, index):
        fileID = self.filelist[index]
        s1_path = os.path.join(self.root, fileID[1], fileID[4])
        s2_cloudfree_path = os.path.join(self.root, fileID[2], fileID[4])
        s2_cloudy_path = os.path.join(self.oroot, fileID[3], fileID[4])

        s1_data = self.get_sar_image(s1_path).astype('float32')
        s2_cloudfree_data = self.get_opt_image(s2_cloudfree_path).astype('float32')
        s2_cloudy_data = self.get_opt_image(s2_cloudy_path).astype('float32')

        if self.use_cloudmask:
            cloud_mask = get_cloud_cloudshadow_mask(s2_cloudy_data, self.opt['cloud_threshold'])
            cloud_mask[cloud_mask != 0] = 1
        '''
        for SAR, clip param: [-25.0, -32.5], [0, 0]
                 minus the lower boundary to be converted to positive
                 normalized by clip_max - clip_min, and increase by max_val
        for optical, clip param: 0, 10000
                     normalized by scale
        '''
        s1_data = self.get_normalized_data(s1_data, data_type=1)
        s2_cloudfree_data = self.get_normalized_data(s2_cloudfree_data, data_type=2)
        s2_cloudy_data = self.get_normalized_data(s2_cloudy_data, data_type=3)

        s1_data = torch.from_numpy(s1_data)
        s2_cloudfree_data = torch.from_numpy(s2_cloudfree_data)
        s2_cloudy_data = torch.from_numpy(s2_cloudy_data)
        if self.use_cloudmask:
            cloud_mask = torch.from_numpy(cloud_mask)

        if self.random_crop and self.base_size - self.crop_size > 0:
            if self.phase == 'train':
                y = random.randint(0, np.maximum(0, self.base_size - self.crop_size))
                x = random.randint(0, np.maximum(0, self.base_size - self.crop_size))
            else:
                y = np.maximum(0, self.base_size - self.crop_size)//2
                x = np.maximum(0, self.base_size - self.crop_size)//2
            s1_data = s1_data[...,y:y+self.crop_size,x:x+self.crop_size]
            s2_cloudfree_data = s2_cloudfree_data[...,y:y+self.crop_size,x:x+self.crop_size]
            s2_cloudy_data = s2_cloudy_data[...,y:y+self.crop_size,x:x+self.crop_size]
            if self.use_cloudmask:
                cloud_mask = cloud_mask[y:y+self.crop_size,x:x+self.crop_size]

        results = {'opt_cloudy': s2_cloudy_data,
                   'opt_clear': s2_cloudfree_data,
                   'sar': s1_data,
                   'file_name': fileID[4]}
        if self.use_cloudmask:
            results['mask'] = cloud_mask

        return results

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_images

    def get_opt_image(self, path):

        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts

        return image

    def get_sar_image(self, path):

        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts

        return image

    def get_normalized_data(self, data_image, data_type):
        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
            data_image /= self.scale

        return data_image
'''
read data.csv
'''
def get_train_val_test_filelists(listpath):

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

    return train_filelist, val_filelist, test_filelist

if __name__ == "__main__":
    ##===================================================##
    parser=argparse.ArgumentParser()
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='../data')
    parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--use_cloudmask', type=bool, default=True)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)
    opts = parser.parse_args() 

    ##===================================================##
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    ##===================================================##
    train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)

    ##===================================================##
    data = AlignedDataset(opts, test_filelist)
    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=1,shuffle=False)

    ##===================================================##
    _iter = 0
    for results in dataloader:
        cloudy_data = results['cloudy_data']
        cloudfree_data = results['cloudfree_data']
        SAR = results['SAR_data']
        if opts.use_cloudmask:
            cloud_mask = results['cloud_mask']
        file_name = results['file_name']
        print(_iter, file_name)
        print('cloudy:', cloudy_data.shape)
        print('cloudfree:', cloudfree_data.shape)
        print('sar:', SAR.shape)
        if opts.use_cloudmask:
            print('cloud_mask:', cloud_mask.shape)
        _iter += 1
