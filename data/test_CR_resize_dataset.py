import os
import random
import numpy as np
import cv2
from torch.utils.data import Dataset

from data.data_util import get_filelists_from_csv
from utils import FileClient, imfrombytes, img2tensor, padding

import torch
import argparse
from utils.parser_option import parse_option
from utils.misc import scandir

class Test_CR_resize_Dataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.root = opt['root']
        self.phase = opt['phase']
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.use_cloudmask =  opt['use_cloudmask'] if 'use_cloudmask' in opt else False
        self.use_id = opt['use_id'] if 'use_id' in opt else False

        self.base_size = opt['base_size']

        if 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            self.filelist = get_filelists_from_csv(opt['meta_info'], self.phase)
        else:
            self.filelist = sorted(list(
                scandir(os.path.join(self.root,  'opt_cloudy'), full_path=False)))

        self.n_images = len(self.filelist)

    # TODO： augmentation for training
    def __getitem__(self, index):
        fileID = self.filelist[index]
        if 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            sar_VH_path = os.path.join(self.root, fileID[1], 'VH', fileID[4].replace('S2', 'S1'))
            sar_VV_path = os.path.join(self.root, fileID[1], 'VV', fileID[4].replace('S2', 'S1'))
            cloudy_path = os.path.join(self.root, fileID[3], fileID[4])
            file_name = fileID[4]
            image_id = int(fileID[-1])
        else:
            sar_VH_path = os.path.join(self.root, 'SAR', 'VH', fileID.replace('S2', 'S1'))
            sar_VV_path = os.path.join(self.root, 'SAR', 'VV', fileID.replace('S2', 'S1'))
            cloudy_path = os.path.join(self.root, 'opt_cloudy', fileID)
            file_name = fileID
            image_id = int(file_name.split('.')[0].split('_')[1])

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        sar_VH_bytes = self.file_client.get(sar_VH_path, 'sar_VH')
        sar_VV_bytes = self.file_client.get(sar_VV_path, 'sar_VV')
        cloudy_bytes = self.file_client.get(cloudy_path, 'cloudy')

        img_sar_VH = imfrombytes(sar_VH_bytes, flag='grayscale', float32=True)
        img_sar_VV = imfrombytes(sar_VV_bytes, flag='grayscale', float32=True)
        img_sar = np.concatenate((img_sar_VH, img_sar_VV), axis=2)

        img_cloudy = imfrombytes(cloudy_bytes, float32=True)


        if self.use_cloudmask:
            mask_path = None
            mask_bytes = self.file_client.get(mask_path, 'mask')
            img_mask = imfrombytes(mask_bytes, flag='grayscale', float32=True)

        # Todo：augmentation for training
        img_cloudy = cv2.resize(img_cloudy, (self.base_size, self.base_size))
        img_sar = cv2.resize(img_sar, (self.base_size, self.base_size))

        img_sar, img_cloudy = img2tensor([img_sar, img_cloudy],
                                    bgr2rgb=True,
                                    float32=True)
        if self.use_cloudmask:
            img_mask = img2tensor(img_mask, float32=True)

        #  normalize
        # mean
        results = {'sar': img_sar,
                   'opt_cloudy':img_cloudy,
                   'file_name': file_name}
        if self.use_cloudmask:
            results['cloud_mask'] = img_mask

        if self.use_id:
            image_id = torch.tensor(image_id, dtype=torch.float32)
            results['image_id'] = image_id

        return results

    def __len__(self):
        return len(self.filelist)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='../options/basic_test_config.yml',
                        help='the path of options file.')
    args = parser.parse_args()
    opt = parse_option(args.opt)
    return opt

# if __name__ == "__main__":
#     opt = parse_args()
#     opt_test = opt['datasets']['test']
#     opt_test['root'] = r'D:\Dataset\Rsipac\test_256'
#
#     dataset = Test_CR_Dataset(opt_test)
#     print(len(dataset))
#     dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=False)
#     _iter = 0
#     for results in dataloader:
#         cloudy_data = results['opt_cloudy']
#         SAR = results['sar']
#         if opt_test['use_cloudmask']:
#             cloud_mask = results['cloud_mask']
#         file_name = results['file_name']
#         print(_iter, file_name)
#         print('cloudy:', cloudy_data.shape)
#         print('sar:', SAR.shape)
#         if opt_test['use_cloudmask']:
#             print('cloud_mask:', cloud_mask.shape)
#         _iter += 1
#         if _iter>2:
#             break