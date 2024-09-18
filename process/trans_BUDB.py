import random
# 写txt
import os
import tqdm

root_path = '../data/BUDB/'
imgpath = '../data/BUDB/images'

image_list = os.listdir(imgpath)

for i in range(5):
    fold_name = root_path + 'fold_' + str(i) + '/'
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)

    train_txt = fold_name + 'train.txt'
    val_txt = fold_name + 'val.txt'

    open_train_txt = open(train_txt, 'w')
    open_val_txt = open(val_txt, 'w')

    # 将训练集图片名称写入txt
    n = 0
    val_txt_len_min = i * 0.2 * len(image_list)
    val_txt_len_max = (i + 1) * 0.2 * len(image_list)
    for img in tqdm.tqdm(image_list):
        if val_txt_len_min <= n < val_txt_len_max:
            name = img[0:-4] + '\n'
            open_val_txt.write(name)
            n = n + 1
        else:
            name = img[0:-4] + '\n'
            open_train_txt.write(name)
            n = n + 1
    open_train_txt.close()
    open_val_txt.close()
