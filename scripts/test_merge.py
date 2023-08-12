import os
import cv2
import numpy as np
from tqdm import tqdm

experiment_name = 'NAF_CR_Middle_crop_2:1/'
refer_path = '../Dataset/Rsipac/test/opt_cloudy/'
infer_path = f'./infer/{experiment_name}/pred_256/'
result_path = f'./infer/{experiment_name}/results_gap/'

if not os.path.exists(result_path):
    os.makedirs(result_path)

target_size = 512
slice_size = 256
overlap_rate = 0.5
gap = slice_size * overlap_rate

image_sets = [i for i in os.listdir(refer_path) if i.endswith(".png")]
print(len(image_sets))

for image_name in tqdm(image_sets):
    result_png = np.zeros((target_size, target_size, 3), np.uint8)
    w, h = target_size, target_size

    x1_s = range(0, w, int(slice_size * (1 - overlap_rate)))
    y1_s = range(0, h, int(slice_size * (1 - overlap_rate)))
    cnt_output = 0
    for y1 in y1_s:
        for x1 in x1_s:
            if x1 + slice_size > w or y1 + slice_size > h:
                continue
            res_y1, res_x1 = y1, x1
            res_y2, res_x2 = y1 + slice_size, x1 + slice_size
            inf_y1, inf_x1 = 0, 0
            inf_y2, inf_x2 = slice_size, slice_size


            if res_y1!=0:
                res_y1 += int(gap//2)
                inf_y1 += int(gap // 2)
            if res_y2!=h:
                res_y2 -= int(gap//2)
                inf_y2 -= int(gap // 2)
            if res_x1!=0:
                res_x1 += int(gap//2)
                inf_x1 += int(gap//2)
            if res_x2!=w:
                res_x2 -= int(gap//2)
                inf_x2 -= int(gap//2)

            infer_name = f'{image_name.split(".")[0]}_{cnt_output}.png'
            assert os.path.isfile(os.path.join(infer_path, infer_name)), f'ERROR: --image {infer_name} does not exist'
            infer_image = cv2.imread(os.path.join(infer_path, infer_name))

            result_png[res_y1:res_y2, res_x1:res_x2,:] = infer_image[inf_y1:inf_y2, inf_x1:inf_x2,:]
            cnt_output += 1

    cv2.imwrite(os.path.join(result_path, image_name), result_png)

