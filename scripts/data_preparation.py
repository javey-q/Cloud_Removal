import os
import csv
from sklearn import model_selection

data_root = r"D:\Dataset\Rsipac\train"
image_sets = os.listdir(os.path.join(data_root, 'opt_clear'))

images_train, images_valid = model_selection.train_test_split(image_sets, test_size=0.2, random_state=0)

#  注意newline=''   否则默认为换行
with open(r"C:\Projects\Datasets\Rsipac\train\train_val_list.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    for image_name in image_sets:
        if image_name in images_train:
            data = [1, 'SAR', 'opt_clear', 'opt_cloudy', image_name]
        else:
            data = [2, 'SAR', 'opt_clear', 'opt_cloudy', image_name]
        writer.writerow(data)


