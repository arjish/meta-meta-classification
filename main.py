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
    help='set for test, otherwise train')
parser.add_argument('--multi', action='store_true', default=False,
    help='set for multi-class problems, otherwise binary classification')
parser.add_argument('-l', '--train_lr', default=1e-4, type=float,
    help='train_lr (default=1e-4)')
parser.add_argument('-p', '--pkl_file', default='filelist',  type=str,
    help='path to pickle file')
parser.add_argument('-cf', '--cluster_folder', default=None, type=str,
    help='cluster folder w/o root (default=None)')

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

parser.add_argument('--iter', default=40000, type=int,
    help='# of training iterations (default=40,000)')
parser.add_argument('--train_problems', default=100000, type=int,
    help='# of training problems (default=100,000)')
parser.add_argument('--test_problems', default=600, type=int,
    help='# of test problems (default=600)')
parser.add_argument('-c', '--cuda_id', default="0", type=str,
    help='cuda ID (default="0")')

args = parser.parse_args()

data_path = args.data_path
data_source = args.data_source
ckpt_name = args.ckpt_name
cluster_folder = args.cluster_folder
train_lr = args.train_lr
pkl_file = args.pkl_file
kshot = args.kshot
kquery = args.kquery
nway = args.nway
meta_batchsz = args.metabatch
steps = args.steps
n_iterations = args.iter
train_problems = args.train_problems
test_problems = args.test_problems
cuda_id = args.cuda_id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

def train(model, saver, sess):
    """

    :param model:
    :param saver:
    :param sess:
    :return:
    """
    # write graph to tensorboard
    # tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)
    prelosses, postlosses, supportaccs, preaccs, postaccs = [], [], [], [], []
    best_acc = 0

    # train for meta_iteartion epoches
    for iteration in range(n_iterations):
        # this is the main op
        ops = [model.meta_op]

        # add summary and print op
        if iteration % 200 == 0:
            ops.extend([model.summ_op,
                        model.query_losses[0], model.query_losses[-1],
                        model.query_accs[0], model.query_accs[-1]
                        ])

        # run all ops
        result = sess.run(ops)

        # summary
        if iteration % 200 == 0:
            # summ_op
            # tb.add_summary(result[1], iteration)
            # query_losses[0]
            prelosses.append(result[2])
            # query_losses[-1]
            postlosses.append(result[3])
            # query_accs[0]
            preaccs.append(result[4])
            # query_accs[-1]
            postaccs.append(result[5])
            # support_acc

            print(iteration, '\tloss:', np.mean(prelosses), '=>', np.mean(postlosses),
                  '\t\tacc:', np.mean(preaccs), '=>', np.mean(postaccs))
            prelosses, postlosses, preaccs, postaccs, supportaccs = [], [], [], [], []

        # evaluation
        if iteration % 2000 == 0:
            # DO NOT write as a = b = [], in that case a=b
            # DO NOT use train variable as we have train func already.
            acc0s, acc1s, acc2s = [], [], []
            # sample 20 times to get more accurate statistics.
            for _ in range(20):
                acc1, acc2 = sess.run([
                                    model.test_query_accs[0],
                                    model.test_query_accs[-1]])
                acc1s.append(acc1)
                acc2s.append(acc2)

            acc = np.mean(acc2s)
            print('>>>>\t\tValidation accs::\t ', np.mean(acc1s), acc, 'best:', best_acc, '\t\t<<<<')

            if acc - best_acc > 0.0:
                saver.save(sess, os.path.join(ckpt_name, 'maml.mdl'))
                best_acc = acc
                print('saved into ckpt:', acc)


def test(model, sess):
    np.random.seed(1)
    random.seed(1)

    # repeat test accuracy for 600 times
    test_accs = []
    for i in range(test_problems):
        if i % 100 == 1:
            print(i)
        # extend return None!!!
        ops = [model.query_preds_probs, model.test_support_acc]
        ops.extend(model.test_query_accs)
        result = sess.run(ops)
        test_accs.append(result[1:])

    # [600, steps+1]
    test_accs = np.array(test_accs)
    # [steps+1]
    means = np.mean(test_accs, 0)
    stds = np.std(test_accs, 0)
    ci95 = 1.96 * stds * 100 / np.sqrt(test_problems)

    print('[support_t0, query_t0 - \t\t\tsteps] ')
    print('mean:', means)
    print('stds:', stds)
    print('ci95:', ci95)


def main():
    training = not args.test
    multiclass = args.multi
    # kshot + kquery images per category, nway categories, meta_batchsz tasks.
    db = DataGenerator(data_source, nway, kshot, kquery, meta_batchsz,
        pkl_file, data_path, cluster_folder, multiclass, train_problems, test_problems)

    if training:  # only construct training model if needed
        # get the tensors
        image_tensor, label_tensor = db.make_data_tensor(training=True)

        support_x = tf.slice(image_tensor, [0, 0, 0], [-1, nway, -1], name='support_x')
        query_x = tf.slice(image_tensor, [0, nway, 0], [-1, -1, -1], name='query_x')
        support_y = tf.slice(label_tensor, [0, 0, 0], [-1, nway, -1], name='support_y')
        query_y = tf.slice(label_tensor, [0, nway, 0], [-1, -1, -1], name='query_y')

    # construct test tensors
    image_tensor, label_tensor = db.make_data_tensor(training=False)
    support_x_test = tf.slice(image_tensor, [0, 0, 0], [-1, nway, -1], name='support_x_test')
    query_x_test = tf.slice(image_tensor, [0, nway, 0], [-1, -1, -1], name='query_x_test')
    support_y_test = tf.slice(label_tensor, [0, 0, 0], [-1, nway, -1], name='support_y_test')
    query_y_test = tf.slice(label_tensor, [0, nway, 0], [-1, -1, -1], name='query_y_test')

    # 1. construct MAML model
    model = MAML(data_source, 2, kshot, kquery, train_lr=train_lr)

    # construct metatrain_ and metaval_
    if training:
        model.build(support_x, support_y, query_x, query_y,
            steps, meta_batchsz, mode='train')
        model.build(support_x_test, support_y_test, query_x_test,
            query_y_test, steps, meta_batchsz, mode='eval')
    else:
        model.build(support_x_test, support_y_test, query_x_test,
            query_y_test, steps, meta_batchsz, mode='test')

    model.summ_op = tf.summary.merge_all()
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

    if training:
        train(model, saver, sess)
    else:
        test(model, sess)


if __name__ == "__main__":
    main()
