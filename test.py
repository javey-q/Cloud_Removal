import json
import os
import time
import random
import numpy as np
import argparse
from datetime import datetime
import cv2

import torch
import accelerate
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
    parser.add_argument('-a', '--accelerator', action='store_true',
                        help='use accelerator')
    args = parser.parse_args()
    opt = parse_option(args.opt)
    opt['device'] = args.device
    opt['accelerator'] = args.accelerator
    return opt


def main():
    begin = time.time()
    opt = parse_args()
    device = torch.device(opt['device'])
    if opt['accelerator']:
        accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False)

    if not os.path.exists(opt['infer_dir']):
        os.makedirs(os.path.join(opt['infer_dir']))
    net = get_arch(opt['network']).to(device).half()  # network
    net.eval()

    test_loader = getLoader(opt['datasets']['test'])
    dataset_name = opt['datasets']['test']['name']
    if  'checkpoint' in opt['Experiment'] and opt['Experiment']['checkpoint']:
        #
        ckpt_path = opt['Experiment']['checkpoint']
        checkpoint = torch.load(ckpt_path, map_location=device)
        # net.load_state_dict(checkpoint['model'])
        load_model_compile(net, checkpoint['model'])
        print(f'load checkpoint from {ckpt_path}')
    else:
        raise AttributeError('checkpoint is needed')

    if opt['accelerator']:
        net, test_loader = accelerator.prepare(net, test_loader)


    use_id = opt['use_id'] if 'use_id' in opt else False
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            if  opt['accelerator']:
                image = batch['opt_cloudy']
                sar = batch['sar']
            else:
                image = batch['opt_cloudy'].to(device).half()
                sar = batch['sar'].to(device).half()

            img_name = batch['file_name']
            if use_id:
                if opt['accelerator']:
                    image_id = batch['image_id']
                    pred = net(image, sar, image_id, accelerator).detach().cpu()
                else:
                    image_id = batch['image_id'].to(device).half()
                    pred = net(image, sar, image_id).detach().cpu()
            else:
                pred = net(image, sar).detach().cpu()

            pred_list = list(torch.split(pred, 1, dim=0))
            save_img = tensor2img(pred_list, rgb2bgr=True)

            for i, img in enumerate(save_img):
                save_img_path = os.path.join(opt['infer_dir'], img_name[i])
                # save_img = cv2.resize(save_img, (target_size, target_size))
                imwrite(img, save_img_path)


if __name__ == '__main__':
    begin_time = time.time()
    main()
    print(time.time() - begin_time)



