import json
import os
import yaml
import re

def parse_option(dir_option: str):
    if not os.path.exists(dir_option):
        raise FileNotFoundError(dir_option)

    with open(dir_option, mode='r', encoding='utf-8') as file:
        # opt = yaml.load(file, Loader=yaml.FullLoader)
        opt = yaml.unsafe_load(file)

    opt['__option_dir__'] = dir_option
    # opt = parse_variable(opt)
    return opt

# def parse_variable(opt):
#     print(opt)
#     opt_new = opt
#     pattern = '\${(.*?)\}'
#     matches = re.findall(pattern, opt)
#     matches = set([m.strip() for m in matches])
#     if matches:
#         for m in matches:
#             if m not in opt:
#                 raise Exception(f'{m} not found in yml data')
#             opt_new = opt_new.replace(f'${{{m}}}', opt[m])
#     print(opt_new)
#     return opt_new

if __name__ == '__main__':
    opt_path = '../options/basic_config.yml'
    opt = parse_option(opt_path)
    print(opt)