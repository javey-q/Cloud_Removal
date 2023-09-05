import json
import os
import time
import random
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import cv2

import torch
import torch.nn
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from arch import get_arch
from data import getLoader

from utils import *
from utils.parser_option import parse_option
from utils.misc import get_latest_run, set_random_seed
from utils.metrics import *

# python valid.py --opt ./options/NAF_refine_crop_config.yml --phase val --visual Contrast --filter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/basic_config.yml',
                        help='the path of options file.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--phase', type=str, default='val',
                        help='phase of dataset: train or val')
    parser.add_argument('--visual', type=str, default='None',
                        help='visual mode: None, Pure, Contrast')
    parser.add_argument('--filter', action='store_true',
                        help=' filter')
    args = parser.parse_args()
    opt = parse_option(args.opt)
    opt['device'] = args.device
    opt['phase'] = args.phase
    opt['visual'] = args.visual
    opt['filter'] = args.filter
    return opt


def main():
    opts = parse_args()
    device = torch.device(opts['device'])

    net = get_arch(opts['network_g']).to(device)  # network
    net.eval()

    data_phase = opts['phase']
    #
    # if data_phase == 'val':
    #     opts['datasets'][data_phase]['batch_size'] = 1
    valid_loader = getLoader(opts['datasets'][data_phase])
    dataset_name = opts['datasets'][data_phase]['name']
    meta_info = opts['datasets'][data_phase]['meta_info']

    save_dir = os.path.join(opts['Experiment']['result_dir'], data_phase + '_infer')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Todo
    if 'new' in meta_info:
        meta_csv = pd.read_csv(meta_info, names=['phase', 'SAR', 'opt_clear', 'opt_cloudy', 'img_name', 'ssim', 'rank'])
    else:
        meta_csv = pd.read_csv(meta_info, names=['phase','SAR', 'opt_clear', 'opt_cloudy','img_name'])

    if  'checkpoint_dir' in opts['Experiment'] and opts['Experiment']['checkpoint_dir']:
        #
        ckpt_dir = opts['Experiment']['checkpoint_dir']
        ckpt_path = os.path.join(ckpt_dir, 'checkpoint_best.pth')
        checkpoint = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(checkpoint['model'])

        print(f'load checkpoint from {ckpt_path}')
    else:
        raise AttributeError('checkpoint is needed')

    m_ssim = AverageMeter('SSIM', ':6.2f', Summary.AVERAGE)
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    end = time.time()
    if opts['filter']:
        if 'ssim' not in list(meta_csv.columns):
            meta_csv['ssim'] = None
    use_gray = opts['network_g']['use_gray'] if 'use_gray' in opts['network_g'] else False
    with torch.no_grad():
        with tqdm(total=len(valid_loader),
                  desc=f'Test on {dataset_name}', unit='batch') as test_pbar:
            for step, batch in enumerate(valid_loader):
                image = batch['opt_cloudy'].to(device)
                sar = batch['sar'].to(device)
                label = batch['opt_clear'].to(device)
                img_name = batch['file_name']

                if use_gray:
                    pred, pred_gray = net(image, sar)
                else:
                    pred = net(image, sar)
                pred_list = torch.split(pred, 1, dim=0)
                label_list = torch.split(label, 1, dim=0)
                for i, (pred, label) in enumerate(zip(pred_list, label_list)):
                    ssim = SSIM(pred, label).item()
                    if opts['filter']:
                        # to do
                        meta_csv.loc[meta_csv.img_name==img_name[i], 'ssim'] = ssim
                    m_ssim.update(ssim, 1)
                # to do
                save_img_path = os.path.join(save_dir, img_name[0])
                if opts['visual'] == 'Pure':
                    save_img = tensor2img([pred], rgb2bgr=True)
                    imwrite(save_img, save_img_path)
                elif opts['visual'] == 'Contrast':
                    img_sample = torch.cat([image.data, pred.data, label.data], -1)  # 按宽拼接
                    grid = make_grid(img_sample, nrow=1, normalize=True)  # 每一行显示的图像列数
                    save_image(grid, save_img_path)

                batch_time.update(time.time() - end)
                end = time.time()

                test_pbar.set_postfix(
                    ordered_dict={'ssim': m_ssim.avg,'batch_time': batch_time.avg})
                test_pbar.update()
    if opts['filter']:
        meta_csv['rank'] = meta_csv.groupby('phase')['ssim'].rank(method='min', ascending=False)
        phase = 1 if data_phase == 'train' else 2
        print(meta_csv.loc[meta_csv.phase == phase, 'ssim'].describe())
        if 'new' not in meta_info:
            meta_info = meta_info.replace('.csv', '_new.csv')
        meta_csv.to_csv(meta_info, header=0, index=0)

    print(f'The average SSIM of dataset {dataset_name} is {m_ssim.avg}.')


if __name__ == '__main__':
    main()



