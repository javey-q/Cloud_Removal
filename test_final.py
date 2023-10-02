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
from torchvision.utils import make_grid
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


def split_tensor(ts):
    bs, _, _, _ = ts.shape
    outs = []
    for i in range(bs):
        ts_tmp = ts[i].unsqueeze(0)
        top_half, bottom_half = torch.chunk(ts_tmp, 2, dim=2)
        a1, a2 = torch.chunk(top_half, 2, dim=3)
        a3, a4 = torch.chunk(bottom_half, 2, dim=3)
        outs.extend([a1, a2, a3, a4])
    return torch.cat(outs, dim=0)


def convert_tensor(ts):
    ts_l = torch.split(ts, 4, dim=0)
    assert ts_l[0].shape == (4, 3, 256, 256), ts_l[0].shape
    out = []
    for idx, ts_tmp in enumerate(ts_l):
        tmp = make_grid(ts_tmp, nrow=2, padding=0)
        assert tmp.shape == (3, 512, 512)
        out.append(tmp)
    return out


def main():
    opt = parse_args()
    device = torch.device(opt['device'])

    if not os.path.exists(opt['infer_dir']):
        os.makedirs(os.path.join(opt['infer_dir']))

    net = get_arch(opt['network']).to(device).half() # network
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

    use_id = opt['use_id'] if 'use_id' in opt else False
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            image = batch['opt_cloudy']
            sar = batch['sar']
            img_name = batch['file_name']
            if use_id:
                image_id = batch['image_id'].repeat_interleave(4)
                image_id = image_id.to(device).half()
            image = split_tensor(image)
            sar = split_tensor(sar)

            image = image.to(device).half()
            sar = sar.to(device).half()

            if use_id:
                pred = net(image, sar, image_id).detach().cpu()
            else:
                pred = net(image, sar).detach().cpu()

            # pred_list = convert_tensor(pred)

            pred_list = list(torch.split(pred, 4, dim=0))

            save_img = tensor2img(pred_list, rgb2bgr=True)
            for i, img in enumerate(save_img):
                save_img_path = os.path.join(opt['infer_dir'], img_name[i])
                imwrite(img, save_img_path)


if __name__ == '__main__':
    begin_time = time.time()
    main()
    print(time.time() - begin_time)
