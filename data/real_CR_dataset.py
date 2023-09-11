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
from data.transforms import paired_random_crop, augment

class Real_CR_Dataset(Dataset):
    def __init__(self, opt):
        super(Real_CR_Dataset, self).__init__()

        self.opt = opt

        self.root = opt['root']
        self.phase = opt['phase']
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.repeat = opt['repeat'] if 'repeat' in opt else 1

        self.random_crop = opt['random_crop'] if 'random_crop' in opt else False
        self.use_cloudmask =  opt['use_cloudmask'] if 'use_cloudmask' in opt else False
        self.use_id =  opt['use_id'] if 'use_id' in opt else False
        self.use_flip =  opt['use_flip'] if 'use_flip' in opt else False
        self.use_rot = opt['use_rot'] if 'use_rot' in opt else False

        self.base_size = opt['base_size']
        if self.random_crop:
            self.crop_size = opt['crop_size']
        self.ssim_filter = opt['ssim_filter'] if 'ssim_filter' in opt else False
        self.weighted_sampler = opt['weighted_sampler'] if 'weighted_sampler' in opt else None

        if 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            filelist = get_filelists_from_csv(opt['meta_info'], self.phase)
            if self.ssim_filter:
                filter_filelist = []
                for file in filelist:
                    # ssim
                    if float(file[5]) >= self.ssim_filter:
                        filter_filelist.append(file)
                self.filelist = filter_filelist
                print('filter list num: ', len(filter_filelist))
            else:
                self.filelist = filelist
        else:
            # todo  paired_paths_from_folder
            pass

        if self.weighted_sampler != None:
            self.sampler_weights = []
            for data in self.filelist:
                ssim = float(data[5])
                reversal = self.weighted_sampler['reversal'] if 'reversal' in self.weighted_sampler else False
                if reversal:
                    weight = 1 / ssim
                else:
                    weight = ssim if ssim > self.weighted_sampler['sampler_filter'] else 0
                self.sampler_weights.append(weight)

        self.n_images = len(self.filelist)

    # TODO： augmentation for training
    def __getitem__(self, index):
        fileID = self.filelist[index % len(self.filelist)]
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

        # if self.use_cloudmask:
        #     mask_path = None
        #     mask_bytes = self.file_client.get(mask_path, 'mask')
        #     img_mask = imfrombytes(mask_bytes, flag='grayscale', float32=True)

        # random crop
        if self.random_crop and self.base_size - self.crop_size > 0:
            img_sar, img_clear, img_cloudy = paired_random_crop(self.opt, [img_sar, img_clear, img_cloudy], self.crop_size)

        # flip, rotation
        trans_sar, trans_clear, trans_cloudy = augment([img_sar, img_clear, img_cloudy], self.use_flip,
                                  self.use_rot)

        tensor_sar, tensor_clear, tensor_cloudy = img2tensor([trans_sar, trans_clear, trans_cloudy],
                                    bgr2rgb=True,
                                    float32=True)

        # if self.use_cloudmask:
        #     tensor_mask = img2tensor(trans_mask, bgr2rgb=False, float32=True)
        # if self.use_gray:
            # trans_gray = cv2.cvtColor(trans_clear, cv2.COLOR_BGR2GRAY)
            # trans_gray = np.expand_dims(trans_gray, axis=-1)
            # tensor_gray= img2tensor(trans_gray, bgr2rgb=False, float32=True)

        results = {'sar': tensor_sar,
                   'opt_cloudy':tensor_cloudy,
                   'opt_clear': tensor_clear,
                   'file_name': fileID[4]}
        # if self.use_cloudmask:
        #     results['cloud_mask'] = tensor_mask
        if self.use_id:
            image_id = int(fileID[7])
            image_id = torch.tensor(image_id)
            results['image_id'] = image_id

        return results

    def __len__(self):
        return len(self.filelist) * self.repeat


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
    opt_train['root'] = r'E:/Dataset/Rsipac/train'
    opt_train['meta_info'] = 'E:/Dataset/Rsipac/train/train_val_list.csv'
    opt_train['base_size'] = 512
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