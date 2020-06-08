"""
Usage instructions:
    First download the required dataset
    and put the contents images in <data_source>/images (without the root folder)

    Then, run the following:
    python resize_images.py -ds <data_source>
"""
from PIL import Image
import glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-ds', '--data_source', default='imagenet', type=str, metavar='DS',
    help='data_source')

args = parser.parse_args()
data_source = args.data_source

image_path = os.path.join(data_source, 'images/*/')

all_images = glob.glob(image_path + '*')

i = 0
for image_file in all_images:
    im = Image.open(image_file)
    if data_source is 'omniglot':
        im = im.resize((28,28), resample=Image.LANCZOS)
    else:
        im = im.resize((84, 84), resample=Image.LANCZOS)

    im.save(image_file)
    i += 1

    if i % 200 == 0:
        print(i)
