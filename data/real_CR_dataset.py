import os
import random
import numpy as np
from torch.utils.data import Dataset

from data.data_util import get_filelists_from_csv
from utils import FileClient, imfrombytes, img2tensor, padding

import torch
import argparse
import cv2
from utils.parser_option import parse_option

class Real_CR_Dataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.root = opt['root']
        self.phase = opt['phase']
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.random_crop = opt['random_crop'] if 'random_crop' in opt else False
        self.use_cloudmask =  opt['use_cloudmask'] if 'use_cloudmask' in opt else False
        self.use_gray =  opt['use_gray'] if 'use_gray' in opt else False

        self.base_size = opt['base_size']
        if self.random_crop:
            self.crop_size = opt['crop_size']

        if 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            self.filelist = get_filelists_from_csv(opt['meta_info'], self.phase)
        else:
            # todo  paired_paths_from_folder
            pass

        self.n_images = len(self.filelist)

    # TODO： augmentation for training
    def __getitem__(self, index):
        fileID = self.filelist[index]
        sar_VH_path = os.path.join(self.root, fileID[1], 'VH', fileID[4].replace('S2', 'S1'))
        sar_VV_path = os.path.join(self.root, fileID[1], 'VV', fileID[4].replace('S2', 'S1'))
        clear_path = os.path.join(self.root, fileID[2], fileID[4])
        cloudy_path = os.path.join(self.root, fileID[3], fileID[4])

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        sar_VH_bytes = self.file_client.get(sar_VH_path, 'sar_VH')
        sar_VV_bytes = self.file_client.get(sar_VV_path, 'sar_VV')
        clear_bytes = self.file_client.get(clear_path, 'clear')
        cloudy_bytes = self.file_client.get(cloudy_path, 'cloudy')

        img_sar_VH = imfrombytes(sar_VH_bytes, flag='grayscale', float32=True)
        img_sar_VV = imfrombytes(sar_VV_bytes, flag='grayscale', float32=True)
        img_sar = np.concatenate((img_sar_VH, img_sar_VV), axis=2)

        img_clear = imfrombytes(clear_bytes, float32=True)
        img_cloudy = imfrombytes(cloudy_bytes, float32=True)

        if self.use_cloudmask:
            mask_path = None
            mask_bytes = self.file_client.get(mask_path, 'mask')
            img_mask = imfrombytes(mask_bytes, flag='grayscale', float32=True)

        if self.use_gray:
            img_gray = cv2.cvtColor(img_clear, cv2.COLOR_BGR2GRAY)
            img_gray = np.expand_dims(img_gray, axis=-1)

        # Todo：augmentation for training

        img_sar, img_clear, img_cloudy = img2tensor([img_sar, img_clear, img_cloudy],
                                    bgr2rgb=True,
                                    float32=True)

        if self.use_cloudmask:
            img_mask = img2tensor(img_mask, bgr2rgb=False, float32=True)
        if self.use_gray:
            img_gray= img2tensor(img_gray, bgr2rgb=False, float32=True)

        #  normalize
        # mean
        if self.random_crop and self.base_size - self.crop_size > 0:
            if self.phase == 'train':
                y = random.randint(0, np.maximum(0, self.base_size - self.crop_size))
                x = random.randint(0, np.maximum(0, self.base_size - self.crop_size))
            else:
                y = np.maximum(0, self.base_size - self.crop_size)//2
                x = np.maximum(0, self.base_size - self.crop_size)//2

            img_sar = img_sar[...,y:y+self.crop_size,x:x+self.crop_size]
            img_clear= img_clear[...,y:y+self.crop_size,x:x+self.crop_size]
            img_cloudy = img_cloudy[...,y:y+self.crop_size,x:x+self.crop_size]
            if self.use_cloudmask:
                img_mask = img_mask[y:y+self.crop_size,x:x+self.crop_size]
            if self.use_gray:
                img_gray = img_gray[y:y + self.crop_size, x:x + self.crop_size]


        results = {'sar': img_sar,
                   'opt_cloudy':img_cloudy,
                   'opt_clear': img_clear,
                   'file_name': fileID[4]}
        if self.use_cloudmask:
            results['cloud_mask'] = img_mask
        if self.use_gray:
            results['gray'] = img_gray

        return results

    def __len__(self):
        return len(self.filelist)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='../options/basic_config.yml',
                        help='the path of options file.')
    args = parser.parse_args()
    opt = parse_option(args.opt)
    return opt

if __name__ == "__main__":
    opt = parse_args()
    opt_train = opt['datasets']['train']
    opt_train['root'] = 'C:/Projects/Dataset/Rsipac/train'
    opt_train['meta_info'] = 'C:/Projects/Dataset/Rsipac/train/train_val_list.csv'

    dataset = Real_CR_Dataset(opt_train)
    print(len(dataset))
    print(dataset.random_crop)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    _iter = 0
    for results in dataloader:
        cloudy_data = results['opt_cloudy']
        clear_data = results['opt_clear']
        SAR = results['sar']
        if opt_train['use_cloudmask']:
            cloud_mask = results['cloud_mask']
        file_name = results['file_name']
        print(_iter, file_name)
        print('cloudy:', cloudy_data.shape)
        print('clear:', clear_data.shape)
        print('sar:', SAR.shape)
        if opt_train['use_cloudmask']:
            print('cloud_mask:', cloud_mask.shape)
        _iter += 1
        if _iter>2:
            break