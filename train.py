import json
import os
import random
import numpy as np
import argparse
from datetime import datetime

import accelerate
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import torch
import torch.nn
from torch.optim import lr_scheduler
import wandb
from tqdm import tqdm

from arch import get_arch
from data import getLoaders
from utils import get_optimizer, get_scheduler
from utils.fakeWandbRun import FakeRun
from utils.parser_option import parse_option
from utils.misc import get_latest_run, set_random_seed

from generic_train_test import Generic_train_test
# from models.model_CR_net import ModelCRNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/basic_config.yml',
                        help='the path of options file.')
    args = parser.parse_args()
    opt = parse_option(args.opt)
    return opt

def make_dirs(opt):
    time_mow = datetime.now()
    if 'checkpoint_dir' not in opt or opt['checkpoint_dir'] == None:
        opt['checkpoint_dir'] = f'./experiments/{time_mow}/checkpoints'
    if 'result_dir' not in opt or opt['result_dir'] == None:
        opt['result_dir'] = f'./experiments/{time_mow}/results'

    os.makedirs(opt['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.join(opt['result_dir'], 'train_images'), exist_ok=True)
    os.makedirs(os.path.join(opt['result_dir'], 'valid_images'), exist_ok=True)


VISIBLE_ALL = True
VISIBLE_ACCELERATE_CONFIG = True
VISIBLE_OPTION = True
VISIBLE_NETWORK = True

logger = get_logger(__name__)  # log_level


def main():
    accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False)

    if VISIBLE_ALL and VISIBLE_ACCELERATE_CONFIG:
        logger.info('Accelerate options details:')
        logger.info('===============================================================')
        logger.info(accelerator.state, main_process_only=False) # all process
        logger.info('===============================================================')

    opt = parse_args()
    if VISIBLE_ALL and VISIBLE_OPTION:
        logger.info('Option details:')
        logger.info('===============================================================')
        logger.info(json.dumps(opt, indent=2))
        logger.info('===============================================================')


    net = get_arch(opt['network_g']) # network_g
    if VISIBLE_ALL and VISIBLE_NETWORK:
        logger.info('Network details:')
        logger.info('===============================================================')
        logger.info(net)
        logger.info('===============================================================')

    if accelerator.is_local_main_process:
        make_dirs(opt['Experiment'])
    # seed
    # torch.backends.cudnn.benchmark = True # 搜索最适合的卷积算法，实现加速
    # torch.backends.cudnn.deterministic = True  # 固定卷积算法，保证网络输出不变
    if opt['manual_seed'] is not None:
        set_random_seed(opt['manual_seed'], cudnn_deterministic=False, cudnn_benchmark=True)
        # set_seed(opt['manual_seed'], device_specific=True)  # 官方实现


    optimizer = get_optimizer(opt['train']['optimizer_g'], optim_params=net.parameters())
    scheduler = get_scheduler(opt['train']['scheduler'], optimizer=optimizer)

    # Choose loss function
    # loss_fn = torch.nn.L1Loss()

    train_loader, val_loaders = getLoaders(opt['datasets'])
    datasets_name = [opt['datasets'][ds]['name'] for ds in opt['datasets']]

    run = FakeRun() # 初始化
    if 'resume' in opt['Experiment'] and opt['Experiment']['resume']:
        # Todo
        resume_opt = opt['Experiment']['resume']
        ckpt_dir = opt['Experiment']['checkpoint_dir']
        ckpt_path = os.path.join(ckpt_dir, resume_opt['resume_ckpt']) if 'resume_ckpt' in  resume_opt and resume_opt['resume_ckpt'] \
                                                                            else get_latest_run(ckpt_dir)
        assert os.path.isfile(ckpt_path), 'ERROR: --resume checkpoint does not exist'
        accelerator.print(f'Resuming training from {ckpt_path}')

        # start_epoch = torch.tensor([accelerator.process_index]).to(accelerator.device)
        # metrics_ssim = torch.tensor([accelerator.process_index]).to(accelerator.device)
        # metrics = {'val_loss': np.inf, 'val_ssim': 0, 'val_psnr': 0}
        # if accelerator.is_local_main_process:
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['model'])
        start_epoch = resume_opt['resume_epoch'] if 'resume_epoch' in  resume_opt and resume_opt['resume_epoch'] \
            else checkpoint['epoch']
        if 'resume_addition' in  resume_opt and resume_opt['resume_addition']:
            # modification max epoch
            scheduler.last_epoch = start_epoch
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
        metrics= checkpoint['metrics']

            # metrics_ssim = checkpoint['metrics']['val_ssim']
            # metrics_ssim = torch.tensor(metrics_ssim, dtype=torch.float32).to(accelerator.device)
            # start_epoch = torch.tensor(start_epoch, dtype=torch.int8).to(accelerator.device)

        # accelerate.utils.broadcast(metrics_ssim, 0)
        # accelerate.utils.broadcast(start_epoch, 0)
        #
        # start_epoch = int(start_epoch)
        # metrics_ssim = float(metrics_ssim)
        # metrics['val_ssim'] = metrics_ssim

        accelerator.print(f'Resume epoch is: {start_epoch} resume metrics is:')
        accelerator.print(metrics)

        if accelerator.is_local_main_process:
            wandb.init(project=opt['Project'], config=opt,
                             resume=True, id=resume_opt['resume_wandb'])
    elif 'finetune' in opt['Experiment'] and opt['Experiment']['finetune']:
        finetune_opt = opt['Experiment']['finetune']
        ckpt_path = finetune_opt['finetune_ckpt']
        assert os.path.isfile(ckpt_path), 'ERROR: --finetune checkpoint does not exist'
        accelerator.print(f'Finetune training from {ckpt_path}')

        if accelerator.is_local_main_process:
            checkpoint = torch.load(ckpt_path)
            net.load_state_dict(checkpoint['model'])
            nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            wandb.init(project=opt['Project'], config=opt,
                       name=f'{opt["Experiment"]["name"]} {nowtime}')
        start_epoch = 0
        metrics = {'val_loss': np.inf, 'val_ssim': 0, 'val_psnr': 0}
    else:
        start_epoch = 0
        if accelerator.is_local_main_process:
            nowtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            wandb.init(project=opt['Project'], config=opt,
                             name=f'{opt["Experiment"]["name"]} {nowtime}')
        metrics = {'val_loss': np.inf, 'val_ssim':0, 'val_psnr':0}

    # 相当于 to(accelerator.device)
    net, optimizer, train_loader, scheduler = accelerator.prepare(net, optimizer, train_loader, scheduler)
    val_loaders = [accelerator.prepare(loader) for loader in val_loaders]
    # Todo
    # model = get_model(cfg['model'])

    train_eval = Generic_train_test(opt, accelerator, net, optimizer, scheduler, train_loader, val_loaders, datasets_name, metrics)
    train_eval.train(accelerator, run, start_epoch, opt['train']['epochs'])


if __name__ == '__main__':
    main()
