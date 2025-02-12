import os
import sys
from imageio import imread, imwrite
from skimage.transform import resize

target_dir = sys.argv[1]
if sys.argv[2] == 'omniglot':
    img_size = [28, 28]
else:
    img_size = [84, 84]

_ids = []

for root, dirnames, filenames in os.walk(target_dir):
    for filename in filenames:
        if filename.endswith(('.jpg', '.webp', '.JPEG', '.png', 'jpeg')):
            _ids.append(os.path.join(root, filename))

for i, path in enumerate(_ids):
    img = imread(path)
    print('{}/{} size: {}'.format(i, len(_ids), img.shape))
    imwrite(path, resize(img, img_size))
