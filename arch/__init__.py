from arch.SimpleNet import SimpleNet
from arch.GLF_CR_net import RDN_residual_CR
from arch.NAF_CR_net import BaselineLocalCR

def get_arch(net_cfg):
    if net_cfg['name'] == 'GLF_CR_Net':
        return RDN_residual_CR(net_cfg)
    elif net_cfg['name'] == 'NAF_CR_Net':
        net_cfg.pop('name')
        return BaselineLocalCR(**net_cfg)
    else:
        raise NotImplementedError(f'{net_cfg["name"]} is not implemented!')

