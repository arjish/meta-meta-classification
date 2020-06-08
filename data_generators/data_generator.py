import numpy as np
import os, sys
import random
import tensorflow as tf
import tqdm
import pickle
from copy import deepcopy


def get_images(path, nb_samples=None, shuffle=True, multi_path=False):
    if nb_samples is not None:
        sampler = lambda x: np.random.choice(x, nb_samples)
    else:
        sampler = lambda x: x

    if multi_path:
        images = [os.path.join(item, image) \
                  for item in path \
                  for image in sampler(os.listdir(item))]
    else:
        images = [os.path.join(path, image) \
                  for image in sampler(os.listdir(path))]

    if shuffle:
        random.shuffle(images)
    return images  # list of images


class DataGenerator:
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """

    def __init__(self, data_source, nway, kshot, kquery,
                 meta_batchsz, pkl_file, data_path,
                 cluster_folder, multiclass,
                 train_batch_num, test_batch_num):
        """

        :param kquery:
        :param meta_batchsz:
        :param nway:
        :param kshot:
        :param kquery:
        :param meta_batchsz:
        :param pkl_file:
        :param data_path:
        :param cluster_folder:
        :param multiclass:
        :param train_batch_num:
        :param test_batch_num:
        """

        self.data_source = data_source
        self.nway = nway
        self.kshot = kshot
        self.kquery = kquery
        self.nimg = kshot + kquery
        self.meta_batchsz = meta_batchsz
        self.pkl_file = pkl_file
        self.data_path = data_path
        self.cluster_flag = True if cluster_folder is not None else False
        self.train_batch_num = train_batch_num
        self.test_batch_num = test_batch_num
        self.multiclass = multiclass

        if self.cluster_flag:
            metatrain_folder_cluster = os.path.join(data_path, cluster_folder, 'train')
            metaval_folder_cluster = os.path.join(data_path, cluster_folder, 'test')

            self.metatrain_images_cluster = []
            for root, dirs, files in os.walk(metatrain_folder_cluster):
                self.metatrain_images_cluster.extend([os.path.join(root, f) for f in files])

            self.metaeval_images_cluster = []
            for root, dirs, files in os.walk(metaval_folder_cluster):
                self.metaeval_images_cluster.extend([os.path.join(root, f) for f in files])

        metatrain_folder_whole = os.path.join(data_path, 'train')
        metaval_folder_whole = os.path.join(data_path, 'test')

        self.metatrain_folders_whole = [os.path.join(metatrain_folder_whole, label) \
                                        for label in os.listdir(metatrain_folder_whole) \
                                        if os.path.isdir(os.path.join(metatrain_folder_whole, label)) \
                                        ]

        self.metaval_folders_whole = [os.path.join(metaval_folder_whole, label) \
                                      for label in os.listdir(metaval_folder_whole) \
                                      if os.path.isdir(os.path.join(metaval_folder_whole, label)) \
                                      ]

        if data_source=='omniglot':
            self.imgsz = (28, 28)
            self.dim_input = np.prod(self.imgsz) * 1  # 784
            self.rotations = [0, 90, 180, 270]
        else:
            self.imgsz = (84, 84)
            self.dim_input = np.prod(self.imgsz) * 3  # 21168
            self.rotations = [0]

        print('Whole data metatrain_folders:', self.metatrain_folders_whole[:2])
        print('Whole data metaval_folders:', self.metaval_folders_whole[:2])

    def make_data_tensor(self, training=True):
        """

        :param training:
        :return:
        """

        if training:
            if self.cluster_flag:
                images_cluster = self.metatrain_images_cluster

            folders = self.metatrain_folders_whole
            num_total_batches = self.train_batch_num
            mode = 'train'
        else:
            if self.cluster_flag:
                images_cluster = self.metaeval_images_cluster

            folders = self.metaval_folders_whole
            num_total_batches = self.test_batch_num
            mode = 'test'

        if training and os.path.exists(self.pkl_file):
            with open(self.pkl_file, 'rb') as f:
                all_filenames = pickle.load(f)
                print('load episodes from file, len:', len(all_filenames))

        else:  # test or not existed.
            all_filenames = []
            for _ in tqdm.tqdm(range(num_total_batches), 'generating episodes'):
                if self.multiclass:
                    # Get the target class folder (positive)
                    class_folders = random.sample(folders, self.nway)
                    # Support images
                    all_filenames.extend(get_images(class_folders,
                        nb_samples=self.kshot, shuffle=False, multi_path=True))
                    # Query images
                    all_filenames.extend(get_images(class_folders,
                        nb_samples=self.kquery, shuffle=False, multi_path=True))
                else:
                    # Get the target class folder (positive)
                    if self.cluster_flag:
                        target_image_cluster = random.sample(images_cluster, 1)[0]
                        wnid = target_image_cluster.split('/')[-2]  # list
                        target_class_folder_whole = os.path.join(self.data_path, mode, wnid)
                    else:
                        target_class_folder_whole = random.sample(folders, 1)[0]

                    # len: 3*self.kquery  + self.kshot
                    # Sample self.kquery negative folders (for train: kquery)
                    # Get the negative classes (removing the target)
                    folders_negative = deepcopy(folders)
                    folders_negative.remove(target_class_folder_whole)

                    sampled_folders_negative = np.random.choice(folders_negative, self.nway - 1, replace=True)
                    # Sample 1 image from each of sampled_folders_negative with label 0
                    filenames = get_images(sampled_folders_negative,
                        nb_samples=1, shuffle=False, multi_path=True)

                    if self.cluster_flag:
                        filenames.append(target_image_cluster)
                    else:
                        # Extend by sampling self.nimg images from target_class_folder_whole with label 1
                        filenames.extend(get_images(target_class_folder_whole,
                            nb_samples=self.kshot, shuffle=False, multi_path=False))

                    # Sample self.kquery negative folders (for test: kquery)
                    # Comment to use the same folders_negative
                    sampled_folders_negative = np.random.choice(folders_negative, self.kquery, replace=True)
                    # Sample 1 image from each of sampled_folders_negative with label 0
                    filenames.extend(get_images(sampled_folders_negative,
                        nb_samples=self.kquery, shuffle=False, multi_path=True))

                    # To make balanced positive examples:
                    filenames.extend(get_images(target_class_folder_whole,
                        nb_samples=(self.nway - 1) * self.kquery, shuffle=False, multi_path=False))

                    # make sure the above isn't randomized order
                    all_filenames.extend(filenames)

            if training:  # only save for training.
                with open(self.pkl_file, 'wb') as f:
                    pickle.dump(all_filenames, f)
                    print('save all file list to:', self.pkl_file)
            else:
                with open(self.pkl_file + '_test', 'wb') as f:
                    pickle.dump(all_filenames, f)
                    print('save all file list to:', self.pkl_file + '_test')

        # make queue for tensorflow to read from
        print('creating pipeline ops')
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)

        if self.data_source == 'omniglot':
            image = tf.image.decode_png(image_file)
            image.set_shape((self.imgsz[0], self.imgsz[1], 1))
            image = tf.cast(image, tf.float32) / 255.0
            image = 1.0 - image  # invert
        else:
            image = tf.image.decode_jpeg(image_file, channels=3)
            # tensorflow format: N*H*W*C
            image.set_shape((self.imgsz[0], self.imgsz[1], 3))
            # convert to range(0,1)
            image = tf.cast(image, tf.float32) / 255.0

        # reshape(image, [84*84*3])
        image = tf.reshape(image, [self.dim_input])

        if self.multiclass:
            examples_per_batch = self.nway * self.nimg
        else:
            # To make balanced positive examples:
            examples_per_batch = (self.nway * self.kshot) + (2 * (self.nway - 1) * self.kquery)

        # batch here means batch of meta-learning, including 4 tasks = 4*151
        batch_image_size = self.meta_batchsz * examples_per_batch  # 4* 151

        print('batching images')
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,  # 4*151
            num_threads=self.meta_batchsz,
            capacity=256 + 3 * batch_image_size,  # 256 + 3* 4*151
        )

        all_image_batches, all_label_batches = [], []
        print('manipulating images to be right order')

        # images contains current batch
        if self.multiclass:
            for i in range(self.meta_batchsz):  # 4
                # current task, 80 images
                image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]
                for j in range(self.nway):
                    labels = [0] * examples_per_batch
                    labels[j] = 1
                    labels[self.nway + j * self.kquery: self.nway + (j + 1) * self.kquery] = [1] * self.kquery
                    label_batch = tf.convert_to_tensor(labels)
                    all_image_batches.append(image_batch)
                    all_label_batches.append(label_batch)

        else:
            # Labels: [0]*50, [1]*50, [0]*50, [1]*50
            labels = [0] * (self.nway - 1) * self.kshot
            labels.extend([1] * self.kshot)
            labels.extend([0] * (self.nway - 1) * self.kquery)
            # To make balanced positive examples:
            labels.extend([1] * (self.nway - 1) * self.kquery)
            for i in range(self.meta_batchsz):  # 4
                image_batch = images[i * examples_per_batch:(i + 1) * examples_per_batch]
                label_batch = tf.convert_to_tensor(labels)
                all_image_batches.append(image_batch)
                all_label_batches.append(label_batch)

        # [4, 151, 84*84*3]
        all_image_batches = tf.stack(all_image_batches)
        # [4, 151]
        all_label_batches = tf.stack(all_label_batches)
        # [4, 151, 2]
        all_label_batches = tf.one_hot(all_label_batches, 2)

        print('image_b:', all_image_batches)
        print('label_onehot_b:', all_label_batches)

        return all_image_batches, all_label_batches


def main():
    kshot = 1
    kquery = 1
    nway = 51
    pkl_file = 'filelistILSRV'
    data_path = 'ILSRV'
    data_source = 'imagenet'
    cluster_folder = None
    meta_batchsz = 1
    multiclass = False
    train_problems = 40000
    test_problems = 10000

    db = DataGenerator(data_source, nway, kshot, kquery, meta_batchsz,
        pkl_file, data_path, cluster_folder, multiclass,
        train_problems, test_problems)

    image_tensor, label_tensor = db.make_data_tensor(training=True)


if __name__=="__main__":
    main()
