import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='',
                        help='opt')
    args = parser.parse_args()
    return args

# experiment_name = 'NAF_CR_Base'

if __name__ == '__main__':
    args = parse_args()
    experiment_name = args.opt.split('.')[0]
    print('experiment_name :'+ experiment_name)
    refer_path = '/test/opt_cloudy/'
    infer_path = f'./infer/{experiment_name}/pred/'
    result_path = f'./infer/results/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    target_size = 512
    slice_size = 256
    pos_xy = [(0, 0), (slice_size, 0), (0, slice_size), (slice_size, slice_size)]

    image_sets = [i for i in os.listdir(refer_path) if i.endswith(".png")]
    print(len(image_sets))
    for image_name in tqdm(image_sets):
        result_png = np.zeros((target_size, target_size, 3), np.uint8)
        for i in range(4):
            infer_name = f'{image_name.split(".")[0]}_{i}.png'
            assert os.path.isfile(os.path.join(infer_path, infer_name)), f'ERROR: --image {infer_name} does not exist'
            infer_image = cv2.imread(os.path.join(infer_path, infer_name))
            topleft_x, topleft_y = pos_xy[i]
            result_png[topleft_y:topleft_y+slice_size,
                    topleft_x:topleft_x+slice_size,:] = infer_image

        cv2.imwrite(os.path.join(result_path, image_name), result_png)

