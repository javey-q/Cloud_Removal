import json
import os
import time
import random
import numpy as np
import argparse
from datetime import datetime
import cv2

import torch
import torch.nn
from tqdm import tqdm

from arch import get_arch
from data import getLoader

from utils import *
from utils.parser_option import parse_option
from utils.misc import get_latest_run, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/test/basic_test_config.yml',
                        help='the path of options file.')
    parser.add_argument('--device', default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    opt = parse_option(args.opt)
    opt['device'] = args.device
    return opt


def main():
    opt = parse_args()
    device = torch.device(opt['device'])

    if not os.path.exists(opt['infer_dir']):
        os.makedirs(os.path.join(opt['infer_dir']))

    net = get_arch(opt['network']).to(device)  # network
    net.eval()

    test_loader = getLoader(opt['datasets']['test'])
    dataset_name = opt['datasets']['test']['name']
    if  'checkpoint' in opt['Experiment'] and opt['Experiment']['checkpoint']:
        #
        ckpt_path = opt['Experiment']['checkpoint']
        checkpoint = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(checkpoint['model'])

        print(f'load checkpoint from {ckpt_path}')
    else:
        raise AttributeError('checkpoint is needed')

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    end = time.time()

    use_gray = opt['network']['use_gray'] if 'use_gray' in opt['network'] else False
    with torch.no_grad():
        with tqdm(total=len(test_loader),
                  desc=f'Test on {dataset_name}', unit='batch') as test_pbar:
            for step, batch in enumerate(test_loader):
                image = batch['opt_cloudy'].to(device)
                sar = batch['sar'].to(device)
                img_name = batch['file_name']

                if use_gray:
                    pred, pred_gray = net(image, sar)
                    pred = pred.detach().cpu()
                else:
                    pred = net(image, sar).detach().cpu()

                save_img = tensor2img([pred], rgb2bgr=True)
                save_img_path = os.path.join(opt['infer_dir'], img_name[0])
                # save_img = cv2.resize(save_img, (target_size, target_size))
                imwrite(save_img, save_img_path)

                batch_time.update(time.time() - end)
                end = time.time()

                test_pbar.set_postfix(
                    ordered_dict={'batch_time': batch_time.avg})
                test_pbar.update()

if __name__ == '__main__':
    main()



