from arch.SimpleNet import SimpleNet
from arch.GLF_CR_net import RDN_residual_CR
from arch.NAF_CR_net import NAF_Local_CR
from arch.NAF_Trans_net import NAF_Trans_Net
from arch.NAF_SGFA_CR_net import NAF_SGFA_Local_CR

def get_arch(net_cfg):
    if net_cfg['name'] == 'GLF_CR_Net':
        return RDN_residual_CR(net_cfg)
    elif net_cfg['name'] == 'NAF_CR_Net':
        net_cfg.pop('name')
        return NAF_Local_CR(**net_cfg)
    elif net_cfg['name'] == 'NAF_SGFA_CR_Net':
        net_cfg.pop('name')
        return NAF_SGFA_Local_CR(**net_cfg)
    elif net_cfg['name'] == 'NAF_Trans_Net':
        net_cfg.pop('name')
        return NAF_Trans_Net(**net_cfg)
    else:
        raise NotImplementedError(f'{net_cfg["name"]} is not implemented!')

