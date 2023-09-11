import torch
from enum import Enum
from utils import lr_scheduler as lr_scheduler
from .file_client import FileClient
from .img_util import crop_border, imfrombytes, img2tensor, imwrite, tensor2img, padding
import collections

def load_model_compile(model, origin_dict, strict=True):
    state_dict = collections.OrderedDict()
    torch2_model_prefix = '_orig_mod.'
    offset2 = len(torch2_model_prefix)
    for key, value in origin_dict.items():
        state_dict[key[offset2: len(key)]] = value
    model.load_state_dict(state_dict, strict=strict)

def get_optimizer(optimizer_cfg, optim_params):
    optimizer_type = optimizer_cfg['type']
    kwargs = optimizer_cfg.get('args', {})

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam([{'params': optim_params}], **kwargs)
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW([{'params': optim_params}], **kwargs)
    elif optimizer_type == 'Adamax':
        optimizer = torch.optim.Adamax([{'params': optim_params}], **kwargs)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD([{'params': optim_params}], **kwargs)
    elif optimizer_type == 'ASGD':
        optimizer = torch.optim.ASGD([{'params': optim_params}], **kwargs)
    elif optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop([{'params': optim_params}], **kwargs)
    elif optimizer_type == 'Rprop':
        optimizer = torch.optim.Rprop([{'params': optim_params}], **kwargs)
    else:
        raise NotImplementedError(f'optimizer {optimizer_type} is not supported yet.')
    return optimizer

def get_scheduler(scheduler_cfg, optimizer):
    scheduler_type = scheduler_cfg['type']
    kwargs = scheduler_cfg.get('args', {})

    if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
        scheduler  =  lr_scheduler.MultiStepRestartLR(optimizer, **kwargs)
    elif scheduler_type == 'CosineAnnealingRestartLR':
        scheduler = lr_scheduler.CosineAnnealingRestartLR(optimizer, **kwargs)
    elif scheduler_type == 'TrueCosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type == 'LinearLR':
        scheduler = lr_scheduler.LinearLR(optimizer, **kwargs)
    elif scheduler_type == 'VibrateLR':
        scheduler = lr_scheduler.VibrateLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(
            f'Scheduler {scheduler_type} is not implemented yet.')

    return  scheduler

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display_common(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display(self, batch, accelerator):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        accelerator.print('\t'.join(entries))

    def display_summary_common(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def display_summary(self, accelerator):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        accelerator.print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'