import os
import numpy as np
import argparse
import random
import tensorflow as tf

from data_generators.data_generator import DataGenerator
from models.maml import MAML

parser = argparse.ArgumentParser()
parser.add_argument('data_path', metavar='DATA',
    help='path to data')
parser.add_argument('ckpt_name', metavar='CKPT',
    help='path to checkpoint')
parser.add_argument('-ds', '--data_source', default='imagenet', type=str,
    help='data_source (imagenet or omniglot)')
parser.add_argument('-t', '--test', action='store_true', default=False,
    help='set for test data, otherwise training data')
parser.add_argument('--multi', action='store_true', default=False,
    help='set for multi-class problems, otherwise binary classification')
parser.add_argument('-l', '--train_lr', default=1e-3, type=float,
    help='train_lr (default=1e-3)')
parser.add_argument('-p', '--pkl_file', default='filelist', type=str,
    help='path to pickle file')
parser.add_argument('-cf', '--cluster_folder', default=None, type=str,
    help='cluster folder w/o root (default=None)')
parser.add_argument('-cl', '--num_clusters', default=16, type=int,
    help='# of clusters (default=4)')
parser.add_argument('-m', '--model_id', default="0", type=str,
    help='model ID (default="0")')

# use kshot = 1, kquery = 15, nway = 5 for 5-way one-shot (multi-class)
parser.add_argument('--kshot', default=1, type=int,
    help='# of shots per class (default=1)')
parser.add_argument('--kquery', default=1, type=int,
    help='# of queries per class (default=1)')
parser.add_argument('--nway', default=51, type=int,
    help='# of classes per problem (default=51)')
parser.add_argument('--metabatch', default=4, type=int,
    help='meta batch-size for training (default=4)')
parser.add_argument('--steps', default=5, type=int,
    help='# of gradient steps (default=5)')
parser.add_argument('--train_problems', default=40000, type=int,
    help='# of training problems (default=40,000)')
parser.add_argument('--test_problems', default=10000, type=int,
    help='# of test problems (default=10,000)')
parser.add_argument('-c', '--cuda_id', default="0", type=str,
    help='cuda ID (default="0")')

args = parser.parse_args()

data_path = args.data_path
data_source = args.data_source
ckpt_name = args.ckpt_name
train_lr = args.train_lr
pkl_file = args.pkl_file
cluster_folder = args.cluster_folder
num_clusters = args.num_clusters
model_id = args.model_id

kshot = args.kshot
kquery = args.kquery
nway = args.nway
meta_batchsz = args.metabatch
steps = args.steps
train_problems = args.train_problems
test_problems = args.test_problems
cuda_id = args.cuda_id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id


def get_preds(model, multiclass, sess):
    np.random.seed(1)
    random.seed(1)

    if args.test:
        n_iterations = test_problems // meta_batchsz
    else:
        n_iterations = train_problems // meta_batchsz

    query_acc = []
    query_preds = []
    for i in range(n_iterations):
        ops = [model.test_query_accs, model.query_preds]
        result = sess.run(ops)
        print('Accuracy:', result[0])
        query_acc.extend(list(result[0]))
        query_preds.extend(result[1])

    if multiclass:
        # always do on clusters
        np.save(os.path.join(data_path, 'CLUSTER_' + str(num_clusters),
                                          'queryPredsMulticlass_' + pkl_file + '_cluster'
                                          + str(num_clusters) + '_' + model_id), query_preds)
    else:
        if cluster_folder is None:
            np.save(os.path.join(data_path, 'WHOLE',
                                              'queryPreds_' + pkl_file
                                              + '_model' + model_id), query_preds)
            np.save(os.path.join(data_path, 'WHOLE',
                                              'queryAcc_' + pkl_file
                                              + '_model' + model_id), query_acc)
        else:
            np.save(os.path.join(data_path, 'CLUSTER_' + str(num_clusters),
                                              'queryPreds_' + pkl_file + '_cluster'
                                              + str(num_clusters) + '_' + model_id), query_preds)
            np.save(os.path.join(data_path, 'CLUSTER_' + str(num_clusters),
                                              'queryAcc_' + pkl_file + '_cluster'
                                              + str(num_clusters) + '_' + model_id), query_acc)

    print('****DONE*****')


def main():
    multiclass = args.multi
    # kshot + kquery images per category, nway categories, meta_batchsz tasks.
    db = DataGenerator(data_source, nway, kshot, kquery, meta_batchsz,
        pkl_file, data_path, cluster_folder, multiclass, train_problems, test_problems)

    image_tensor, label_tensor = db.make_data_tensor(training=True)

    # get the tensors
    support_x = tf.slice(image_tensor, [0, 0, 0], [-1, nway, -1], name='support_x')
    query_x = tf.slice(image_tensor, [0, nway, 0], [-1, -1, -1], name='query_x')
    support_y = tf.slice(label_tensor, [0, 0, 0], [-1, nway, -1], name='support_y')
    query_y = tf.slice(label_tensor, [0, nway, 0], [-1, -1, -1], name='query_y')

    model = MAML(data_source, 2, kshot, kquery, train_lr=train_lr)
    model.build(support_x, support_y, query_x, query_y, steps, meta_batchsz, mode='testEach')

    all_vars = filter(lambda x: 'meta_optim' not in x.name, tf.trainable_variables())
    for p in all_vars:
        print(p)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    # tf.global_variables() to save moving_mean and moving variance of batch norm
    # tf.trainable_variables()  NOT include moving_mean and moving_variance.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    # initialize, under interative session
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if os.path.exists(os.path.join(ckpt_name, 'checkpoint')):
        # alway load ckpt both train and test.
        model_file = tf.train.latest_checkpoint(ckpt_name)
        print("Restoring model weights from ", model_file)
        saver.restore(sess, model_file)

    get_preds(model, multiclass, sess)


if __name__=="__main__":
    main()
