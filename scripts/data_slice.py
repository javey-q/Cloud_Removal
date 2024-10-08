import os
import cv2
import csv
from sklearn import model_selection
import pandas as pd
import argparse

# phase = 'train'
# data_root = "../Dataset/Rsipac/train"
# data_slice= "../Dataset/Rsipac/train_320"
# data_csv = "../Dataset/Rsipac/train_320/train_val_list.csv"

# slice_size = 320
# overlap_rate = 0.4

# phase = 'test'
# data_root = r"/root/autodl-tmp/Rsipac/testB"
# data_slice= r"/root/autodl-tmp/Rsipac/testB_256"
# data_csv = r"/root/autodl-tmp/Rsipac/testB_256/train_val_list.csv"
#
# slice_size = 256
# overlap_rate = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train',
                        help='phase')
    parser.add_argument('--data_root', type=str, default='',
                        help='data_root')
    parser.add_argument('--data_slice', type=str, default='',
                        help='data_slice')
    parser.add_argument('--data_csv', type=str, default='',
                        help='data_csv')
    parser.add_argument("--slice_size", type=int, default=256,
                        help="slice_size.")
    parser.add_argument("--overlap_rate", type=float, default=0.,
                        help="overlap_rate.")
    args = parser.parse_args()
    return args

def main(args):
    phase = args.phase
    data_root = args.data_root
    data_slice = args.data_slice
    data_csv = args.data_csv

    slice_size = args.slice_size
    overlap_rate = args.overlap_rate


    if not os.path.exists(data_slice):
        if phase == 'train':
            os.makedirs(os.path.join(data_slice, 'opt_clear'))
        os.makedirs(os.path.join(data_slice, 'opt_cloudy'))
        os.makedirs(os.path.join(data_slice, 'SAR', 'VV'))
        os.makedirs(os.path.join(data_slice, 'SAR', 'VH'))

    def crop_image(image, x1, y1, wh):
        h, w = image.shape[:2]
        if x1 + wh <= w and y1 + wh <= h:
            return image[y1:y1 + wh, x1:x1 + wh], x1, y1
        else:
            print('warning')
            return image[
                   min(y1 + wh, h) - wh:min(y1 + wh, h),
                   min(x1 + wh, w) - wh:min(x1 + wh, w)
                   ], min(x1 + wh,w) - wh, min(y1 + wh, h) - wh

    if phase == 'train':
        origin_list = pd.read_csv(os.path.join(data_root, 'train_val_list.csv'), names=['phase','SAR', 'opt_clear', 'opt_cloudy','img_name'])
        phase_id = 1 if phase == 'train' else 2
        image_sets = origin_list['img_name'].to_list()
        images_train = origin_list.loc[origin_list.phase == 1, 'img_name'].to_list()
        images_valid = origin_list.loc[origin_list.phase == 2, 'img_name'].to_list()
    else:
        image_sets = os.listdir(os.path.join(data_root, 'opt_cloudy'))
        images_train, images_valid = model_selection.train_test_split(image_sets, test_size=0.2, random_state=0)
        print(images_valid)

    with open(data_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        for image_name in image_sets:
            print(image_name)
            if phase == 'train':
                opt_clear_img = cv2.imread(os.path.join(data_root, 'opt_clear', image_name))
            opt_cloudy_img = cv2.imread(os.path.join(data_root, 'opt_cloudy', image_name))
            SAR_VV_img = cv2.imread(os.path.join(data_root, 'SAR', 'VV', image_name.replace('S2', 'S1')), cv2.IMREAD_GRAYSCALE)
            SAR_VH_img = cv2.imread(os.path.join(data_root, 'SAR', 'VH', image_name.replace('S2', 'S1')), cv2.IMREAD_GRAYSCALE)

            h, w, c = opt_cloudy_img.shape
            assert  (h, w, c) == (512, 512, 3), f'image_size: ({h}, {w}, {c})'

            x1s = range(0, w, int(slice_size * (1 - overlap_rate)))
            y1s = range(0, h, int(slice_size * (1 - overlap_rate)))
            cnt_output = 0
            for y1 in y1s:
                for x1 in x1s:
                    if x1 + slice_size > w or y1 + slice_size > h:
                        continue
                    # print(x1, y1, x1+slice_size, y1+slice_size)
                    opt_cloudy_cut, _, _ = crop_image(opt_cloudy_img, x1, y1, slice_size)
                    SAR_VV_cut, _, _ = crop_image(SAR_VV_img, x1, y1, slice_size)
                    SAR_VH_cut, _, _ = crop_image(SAR_VH_img, x1, y1, slice_size)
                    if phase == 'train':
                        opt_clear_cut, _, _ = crop_image(opt_clear_img, x1, y1, slice_size)

                    output_name = f'{image_name.split(".")[0]}_{cnt_output}.png'
                    if phase == 'train':
                        cv2.imwrite(os.path.join(data_slice, 'opt_clear', output_name), opt_clear_cut)
                    cv2.imwrite(os.path.join(data_slice, 'opt_cloudy',output_name), opt_cloudy_cut)
                    cv2.imwrite(os.path.join(data_slice, 'SAR', 'VV', output_name.replace('S2', 'S1')), SAR_VV_cut)
                    cv2.imwrite(os.path.join(data_slice, 'SAR', 'VH', output_name.replace('S2', 'S1')), SAR_VH_cut)
                    cnt_output += 1

                    if phase == 'train':
                        if image_name in images_train:
                            data = [1, 'SAR', 'opt_clear', 'opt_cloudy', output_name]
                        else:
                            data = [2, 'SAR', 'opt_clear', 'opt_cloudy', output_name]
                    else:
                        id = int(image_name.split('_')[1])
                        data = [3, 'SAR', 'opt_clear', 'opt_cloudy', output_name, id]
                    writer.writerow(data)

            assert  cnt_output == 4, f'{cnt_output}!=4'
    print(cnt_output)
    print(len(image_sets))


if __name__ == '__main__':
    args = parse_args()
    main(args)



