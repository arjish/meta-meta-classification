import numpy as np
from PIL import Image
import os

def getData_OvA(path_list, feature_path, meta_batchsz, k_images, positive_class_idx):
    features_support_batch = []

    for i in range(meta_batchsz):
        imagepath = path_list[i * k_images+ positive_class_idx]
        root, filename = os.path.split(imagepath)
        filename = filename.rsplit('.', 1)[0] + '.npy'
        root, wnid = os.path.split(root)

        feature_file = os.path.join(feature_path, wnid, filename)
        #print('support feature_file:', feature_file)
        features_support_batch.append(np.load(feature_file))

    return np.array(features_support_batch)

def getData_nway(path_list, feature_path, nway):
    features_support_batch = []

    support_imagepaths = path_list[: nway]
    for j in range(nway):
        imagepath = support_imagepaths[j]
        root, filename = os.path.split(imagepath)
        filename = filename.rsplit('.', 1)[0] + '.npy'
        root, wnid = os.path.split(root)

        feature_file = os.path.join(feature_path, wnid, filename)
        #print('support feature_file:', feature_file)
        features_support_batch.append(np.load(feature_file))

    return np.array(features_support_batch)

def input_parser(path):
    image_file = path + '.JPEG'
    image = Image.open(image_file)
    image = np.asarray(image)
    # reshape(image, [84*84*3])
    image = np.ravel(image)
    # convert to range(0,1)
    image = image / 255.0
    if image.shape[0] == 84*84:
        image = np.repeat(image, 3)
    # print(image.shape)

    return image


