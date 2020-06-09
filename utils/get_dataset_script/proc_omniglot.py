import os
import random
import errno
import subprocess
from shutil import copytree

source_dir = 'omniglot'
target_dir = 'data/omniglot'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# change folder structure :
alphabet_folders = [family \
                    for family in os.listdir(source_dir) \
                    if os.path.isdir(os.path.join(source_dir, family))]

for folder in alphabet_folders:
    alphabet_path = os.path.join(source_dir, folder)
    char_folders = [character \
                for character in os.listdir(alphabet_path) \
                if os.path.isdir(os.path.join(alphabet_path, character))]

    for character in os.listdir(alphabet_path):
        if os.path.isdir(os.path.join(alphabet_path, character)):
            new_char_path = os.path.join(target_dir, folder + '_' + character)
            try:
                copytree(os.path.join(alphabet_path, character), new_char_path)
            except:
                pass


# train-test split :
character_folders = [os.path.join(target_dir, family, character) \
                for family in os.listdir(target_dir) \
                if os.path.isdir(os.path.join(target_dir, family)) \
                for character in os.listdir(os.path.join(target_dir, family))]

print('Total number of character folders:', len(character_folders))

random.seed(123)
random.shuffle(character_folders)

num_train = 1200

train_folders = character_folders[:num_train]
test_folders = character_folders[num_train:]


if not os.path.exists(os.path.join(target_dir, 'train')):
    os.makedirs(os.path.join(target_dir, 'train'))
    for folder in train_folders:
        root, char_folder = os.path.split(folder)
        root, alphabet_folder = os.path.split(root)
        try:
            copytree(folder, os.path.join(root, 'train', alphabet_folder, char_folder))
        except OSError as e:
            # If the error was caused because the source wasn't a directory, simply ignore
            if e.errno==errno.ENOTDIR:
                pass
            else:
                print('Could not copy directory!')

if not os.path.exists(os.path.join(target_dir, 'test')):
    os.makedirs(os.path.join(target_dir, 'test'))
    for folder in test_folders:
        root, char_folder = os.path.split(folder)
        root, alphabet_folder = os.path.split(root)
        try:
            copytree(folder, os.path.join(root, 'test', alphabet_folder, char_folder))
        except OSError as e:
            # If the error was caused because the source wasn't a directory, simply ignore
            if e.errno == errno.ENOTDIR:
                pass
            else:
                print('Could not copy directory!')

# resize images
cmd = ['python', 'utils/get_dataset_script/resize_dataset.py', 'data/omniglot', 'omniglot']
subprocess.call(cmd)



