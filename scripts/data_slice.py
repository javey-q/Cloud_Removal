import os
import cv2
import csv
from sklearn import model_selection

phase = 'test'
data_root = r"E:\Dataset\Rsipac\test"
data_slice= r"E:\Dataset\Rsipac\test_256_0.5"
data_csv = r"E:\Dataset\Rsipac\test_256_0.5\train_val_list.csv"

slice_size = 256
overlap_rate = 0.5

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
                    writer.writerow(data)

        # assert  cnt_output == 16, f'{cnt_output}!=16'
print(cnt_output)



