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
import accelerate

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


def split_tensor(ts):
    # assert ts.shape == (1, 3, 512, 512), f'tensor shape is {ts.shape}'
    bs, _, _, _ = ts.shape
    outs = []
    for i in range(bs):
        ts_tmp = ts[i].unsqueeze(0)
        top_half, bottom_half = torch.chunk(ts_tmp, 2, dim=2)
        # assert top_half.shape == (1, 3, 256, 512) and bottom_half.shape == (1, 3, 256, 512)
        a1, a2 = torch.chunk(top_half, 2, dim=3)
        a3, a4 = torch.chunk(bottom_half, 2, dim=3)
        # out = torch.cat([a1, a2, a3, a4], dim=0)
        # outs.append(out)
        outs.extend([a1, a2, a3, a4])
    # assert out.shape == (4, 3, 256, 256)
    # return out
    return torch.cat(outs, dim=0)


def convert_tensor(ts):
    ts_l = torch.split(ts, 4, dim=0)
    assert ts_l[0].shape == (4, 3, 256, 256), ts_l[0].shape
    out = []
    for idx, ts_tmp in enumerate(ts_l):
        # slice_size = 256
        # pos_yx = [(0, 0), (0, slice_size), (slice_size, 0), (slice_size, slice_size)]
        # for i in range(4):
        #     topleft_y, topleft_x = pos_yx[i]
        #     out[idx, :, topleft_y:topleft_y+slice_size, topleft_x:topleft_x+slice_size] = ts_tmp[i, :, :, :]
        tmp = make_grid(ts_tmp, nrow=2, padding=0)
        assert tmp.shape == (3, 512, 512)
        out.append(tmp)
    # return out
    return out


def main():
    opt = parse_args()
    device = torch.device(opt['device'])
    if opt['accelerator']:
        accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False)

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
        # net.load_state_dict(checkpoint['model'])
        load_model_compile(net, checkpoint['model'])

        print(f'load checkpoint from {ckpt_path}')
    else:
        raise AttributeError('checkpoint is needed')

    if opt['accelerator']:
        net, test_loader = accelerator.prepare(net, test_loader)
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    begin = time.time()
    end = time.time()

    use_id = opt['use_id'] if 'use_id' in opt else False
    with torch.no_grad():
        with tqdm(total=len(test_loader),
                  desc=f'Test on {dataset_name}', unit='batch') as test_pbar:
            for step, batch in enumerate(test_loader):
                if opt['accelerator']:
                    image = batch['opt_cloudy']
                    sar = batch['sar']
                else:
                    image = batch['opt_cloudy'].to(device)
                    sar = batch['sar'].to(device)

                img_name = batch['file_name']

                image = split_tensor(image)
                sar = split_tensor(sar)

                if use_id:
                    if opt['accelerator']:
                        image_id = batch['image_id'].repeat_interleave(4)
                        pred = net(image, sar, image_id, accelerator).detach().cpu()
                    else:
                        image_id = batch['image_id'].repeat_interleave(4).to(device)
                        pred = net(image, sar, image_id).detach().cpu()
                else:
                    pred = net(image, sar).detach().cpu()

                pred = convert_tensor(pred)
                pred_list = pred

                # pred_list = list(torch.split(pred, 4, dim=0))
                save_img = tensor2img(pred_list, rgb2bgr=True)
                # print(len(pred_list), len(save_img))
                for i, img in enumerate(save_img):
                    save_img_path = os.path.join(opt['infer_dir'], img_name[i])
                    # save_img = cv2.resize(save_img, (target_size, target_size))
                    imwrite(img, save_img_path)

                batch_time.update(time.time() - end)
                end = time.time()

                test_pbar.set_postfix(
                    ordered_dict={'batch_time': batch_time.avg})
                test_pbar.update()
    print(f'Total time:{time.time() - begin}')

if __name__ == '__main__':
    main()


