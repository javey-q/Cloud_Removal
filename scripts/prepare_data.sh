unzip /data/train_训练集（图像+真值）.zip -d /data
rar x /test/test.rar # test_测试集（仅图像）-> testB
cp ./others/train_val_list.csv /data/train
# cut 256
python ./scipts/data_slice.py --phase train --data_root /data/train \
        --data_slice /data/train_256  --data_csv   /data/train_256/train_val_list.csv \
        --slice_size 256 --overlap_rate 0
# cut 320
python ./scipts/data_slice.py --phase train --data_root /data/train \
        --data_slice /data/train_320  --data_csv   /data/train_320/train_val_list.csv \
        --slice_size 320 --overlap_rate 0.4
# cut test
python ./scipts/data_slice.py --phase test --data_root /test/testB \
        --data_slice /test/testB_256  --data_csv   /test/testB_256/test_list.csv \
        --slice_size 256 --overlap_rate 0
cp ./others/train_val_list_new.csv ./data/train_256
cp ./others/train_val_list_new.csv ./data/train_320
cp ./others/test_list.csv /test/testB_256




