import os
import time
import argparse
import multiprocessing

import torch

from arch import get_arch
from data import getLoader

from utils import *
from utils.parser_option import parse_option
from torchvision.utils import make_grid

def split_tensor(ts):
    bs, _, _, _ = ts.shape
    if bs == 1:
        top_half, bottom_half = torch.chunk(ts, 2, dim=2)
        a1, a2 = torch.chunk(top_half, 2, dim=3)
        a3, a4 = torch.chunk(bottom_half, 2, dim=3)
        out = torch.cat([a1, a2, a3, a4], dim=0)
        return out
    else:
        outs = []
        for i in range(bs):
            ts_tmp = ts[i].unsqueeze(0)
            top_half, bottom_half = torch.chunk(ts_tmp, 2, dim=2)
            a1, a2 = torch.chunk(top_half, 2, dim=3)
            a3, a4 = torch.chunk(bottom_half, 2, dim=3)
            outs.extend([a1, a2, a3, a4])
        return torch.cat(outs, dim=0)


def convert_tensor(ts):
    bs, _, _, _ = ts.shape
    if bs == 1:
        slice_size = 256
        pos_xy = [(0, 0), (0, slice_size), (slice_size, 0), (slice_size, slice_size)]
        out = torch.zeros((1, 3, 512, 512))
        for i in range(4):
            topleft_x, topleft_y = pos_xy[i]
            out[0, :, topleft_x:topleft_x+slice_size, topleft_y:topleft_y+slice_size] = ts[i, :, :, :]
        return out
    else:
        ts_l = torch.split(ts, 4, dim=0)
        out = []
        for idx, ts_tmp in enumerate(ts_l):
            tmp = make_grid(ts_tmp, nrow=2, padding=0)
            out.append(tmp)
        return out


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


def inference(sender):
    print('Start')
    
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
        # net.load_state_dict(checkpoint['model'])
        load_model_compile(net, checkpoint['model'])

        print(f'load checkpoint from {ckpt_path}')
    else:
        raise AttributeError('checkpoint is needed')
    
    use_id = opt['use_id'] if 'use_id' in opt else False
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            image = batch['opt_cloudy'].to(device)
            sar = batch['sar'].to(device)
            bs, _, _, _ = image.shape
            
            image = split_tensor(image)
            sar = split_tensor(sar)

            img_name = batch['file_name']
            if use_id:
                image_id = batch['image_id'].to(device)
                if bs > 1:
                    image_id = image_id.repeat_interleave(4)

            if use_id:
                pred = net(image, sar, image_id).detach().cpu()
            else:
                pred = net(image, sar).detach().cpu()

            pred = convert_tensor(pred)

            if bs == 1:
                save_img_path = os.path.join(opt['infer_dir'], img_name[0])
                save_img = tensor2img([pred], rgb2bgr=True)
                sender.send((save_img_path, save_img))
            else:
                save_imgs = tensor2img(pred, rgb2bgr=True)
                for i, save_img in enumerate(save_imgs):
                    save_img_path = os.path.join(opt['infer_dir'], img_name[i])
                    sender.send((save_img_path, save_img))

    sender.send(('over', 0))
    print('Inference finished!')


def save_img_task(receiver):
    while True:
        save_img_path, save_img = receiver.recv()
        if save_img_path == 'over':
            break
        imwrite(save_img, save_img_path)

    print('Save images finished!')


def inference_test(sender):
    time.sleep(2)
    for i in range(50):
        print(f'sending {i}')
        sender.send((i % 2, i))
        time.sleep(0.1)

    sender.send(('over', 0))


def save_img_test(receiver):
    while True:
        val, _ = receiver.recv()
        if val == 'over':
            break
        print(val)


def main():
    sender, receiver = multiprocessing.Pipe(duplex=True)

    inference_process = multiprocessing.Process(
        target=inference,
        args=(sender, )
    )
    save_img_process = multiprocessing.Process(
        target=save_img_task,
        args=(receiver, )
    )

    inference_process.start()
    save_img_process.start()

    inference_process.join()
    save_img_process.join()


if __name__ == '__main__':
    begin_time = time.time()
    main()
    print(time.time() - begin_time)