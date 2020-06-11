import os
import subprocess

source_dir = 'CUB_200_2011/images'
target_dir = 'data/CUB/test'
# We use CUB dataset only for testing

classes = os.listdir(source_dir)

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

cmd = ['mv'] + [os.path.join(source_dir, c) for c in classes] + [target_dir]
subprocess.call(cmd)

# resize images
cmd = ['python', 'utils/get_dataset_script/resize_dataset.py', 'data/CUB/test', 'CUB']
subprocess.call(cmd)
