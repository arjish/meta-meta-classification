import numpy as np
import os
import subprocess

source_dir = 'imagenet/train'
target_dir = 'data/ILSVRC'

percentage_train_class = 90
percentage_test_class = 10
train_test_ratio = [
    percentage_train_class, percentage_test_class]

classes = os.listdir(source_dir)

rs = np.random.RandomState(123)
rs.shuffle(classes)
num_train, num_test = [
    int(float(ratio) / np.sum(train_test_ratio) * len(classes))
    for ratio in train_test_ratio]

classes = {
    'train': classes[:num_train],
    'test': classes[num_train:]
}

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for k in classes.keys():
    target_dir_k = os.path.join(target_dir, k)
    if not os.path.exists(target_dir_k):
        os.makedirs(target_dir_k)
    cmd = ['mv'] + [os.path.join(source_dir, c) for c in classes[k]] + [target_dir_k]
    subprocess.call(cmd)

# move the training classes from imagenet/validation to data/ILSVRC/val (for Resnet training)
source_val_dir = 'imagenet/validation'
target_val_dir = os.path.join(target_dir, 'val')
if not os.path.exists(target_val_dir):
    os.makedirs(target_val_dir)

cmd = ['mv'] + [os.path.join(source_val_dir, c) for c in classes['train']] + [target_val_dir]


# resize images
# cmd = ['python', 'utils/get_dataset_script/resize_dataset.py', 'data/ILSVRC', 'ILSVRC']
# subprocess.call(cmd)