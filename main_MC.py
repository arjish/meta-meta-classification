import os
import numpy as np
import argparse
import random
import tensorflow as tf
import pickle

from models.model_MC import MOE
from utils.processData_moe import getData_OvA, getData_nway

parser = argparse.ArgumentParser()
parser.add_argument('data_path', metavar='DATA',
    help='path to data')
parser.add_argument('ckpt_name', metavar='CKPT',
    help='path to checkpoint')
parser.add_argument('-t', '--test', action='store_true', default=False,
    help='set for test, otherwise train')
parser.add_argument('-w', '--whole', action='store_true', default=False,
    help='set for whole_data, otherwise clusters')
parser.add_argument('--multi', action='store_true', default=False,
    help='set for multi-class problems, otherwise binary classification')
parser.add_argument('-p', '--pkl_file', default='filelist',  type=str,
    help='path to pickle file')
parser.add_argument('-n', '--n_models', default=16, type=int,
    help='# of models (default=16)')

# use kshot = 1, kquery = 15, nway = 5 for 5-way one-shot (multi-class)
parser.add_argument('--kshot', default=1, type=int,
    help='# of shots per class (default=1)')
parser.add_argument('--kquery', default=1, type=int,
    help='# of queries per class (default=1)')
parser.add_argument('--nway', default=51, type=int,
    help='# of classes per problem (default=51)')
parser.add_argument('--metabatch', default=20, type=int,
    help='meta batch-size for training (default=20)')
parser.add_argument('--feature_size', default=512, type=int,
    help='feature size from pre-trained ResNet (default=512)')

parser.add_argument('-c', '--cuda_id', default="0", type=str,
    help='cuda ID (default="0")')
args = parser.parse_args()

ckpt_name = args.ckpt_name
n_models = args.n_models
pkl_file = args.pkl_file
data_path = args.data_path
kshot = args.kshot
kquery = args.kquery
nimg = kshot + kquery
nway = args.nway
meta_batchsz = nway if args.multi else args.metabatch
feature_size = args.feature_size

cuda_id = args.cuda_id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

# define k_images, labels_query
# and load the query_preds:
query_preds_list = []
if args.multi:
    k_images = kshot + kquery
    labels_query = np.arange(nway).repeat(kquery).tolist()

    for i in range(n_models):
        # always do on clusters
        query_preds_list.append(np.load(os.path.join(data_path,
            'CLUSTER_' + str(n_models),
            'queryPredsMulticlass_' + pkl_file + '_cluster'
            + str(n_models) + '_' + str(i) + '.npy')))

else:
    # to make balanced positive examples:
    k_images = (nway * kshot) + (2 * (nway - 1) * kquery)
    labels_query = [0] * ((nway - 1) * kquery)
    labels_query.extend([1] * (nway - 1) * kquery)

    for i in range(n_models):
        if args.whole:
            query_preds_list.append(np.load(os.path.join(data_path, 'WHOLE',
                'queryPreds_' + pkl_file + '_model' + str(i) + '.npy')))
        else:
            query_preds_list.append(np.load(os.path.join(data_path,
                'CLUSTER_' + str(n_models),
                'queryPreds_' + pkl_file + '_cluster'
                + str(n_models) + '_' + str(i) + '.npy')))


def train(model, all_files, saver, sess):
    """

    :param model:
    :param saver:
    :param sess:
    :return:
    """
    # write graph to tensorboard
    # tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)

    # train for meta_iteartion epochs
    n_epochs = 50

    batch_size = meta_batchsz * k_images
    num_batches = len(all_files) // batch_size
    print('Num of batches::', num_batches)
    feature_path = os.path.join(data_path, 'features_train')

    for outer_iter in range(n_epochs):  # this is the main op
        losses, accs = [], []
        for iteration in range(num_batches):
            # start = time.time()
            start_id = (iteration * batch_size)
            end_id = ((iteration + 1) * batch_size)

            ops = [model.updateModel]

            # add summary and print op
            if iteration % 100==0:
                ops.extend([model.query_probs_moe, model.query_loss, model.query_acc])

            # run all ops
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

            query_preds_batch = []
            for i in range(n_models):
                query_preds_batch.append(query_preds_list[i][iteration * meta_batchsz: (iteration + 1) * meta_batchsz])

            features_support_batch = getData_OvA(all_files[start_id:end_id], feature_path,
                meta_batchsz, k_images, kshot * (nway - 1))

            feed_dict = {model.support_f: features_support_batch,
                         model.query_preds: query_preds_batch,
                         model.query_y: [labels_query for k in range(meta_batchsz)],
                         model.keep_prob_in: 0.9,
                         model.keep_prob_hidden: 0.6}
            result = sess.run(ops, feed_dict=feed_dict, options=run_options)
            if iteration % 100==0:
                losses.append(result[2])
                accs.append(result[3])

            # end = time.time()
            # print('Time taken:', end - start)

            if iteration % 400==0:
                print(outer_iter, '\t', iteration, '\tloss:', np.mean(losses),
                    '\t\tacc:', np.mean(accs))

                saver.save(sess, os.path.join(ckpt_name, 'classifier.mdl'))
                print('saved into ckpt!')


def test(model, all_files, sess):
    np.random.seed(1)
    random.seed(1)
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    batch_size = meta_batchsz * k_images
    num_batches = len(all_files) // batch_size
    feature_path = os.path.join(data_path, 'features_test')

    ops = [model.query_acc, model.query_probs_moe]

    test_accs = []
    for iteration in range(num_batches):
        start_id = (iteration * batch_size)
        end_id = ((iteration + 1) * batch_size)

        query_preds_batch = []
        for i in range(n_models):
            query_preds_batch.append(query_preds_list[i][iteration * meta_batchsz:
                                                         (iteration + 1) * meta_batchsz])

        if args.multi:
            features_support_batch = getData_nway(all_files[start_id:end_id],
                feature_path, nway)
        else:
            features_support_batch = getData_OvA(all_files[start_id:end_id], feature_path,
                meta_batchsz, k_images, kshot * (nway - 1))

        feed_dict = {model.support_f: features_support_batch,
                     model.query_preds: query_preds_batch,
                     model.query_y: [labels_query for k in range(meta_batchsz)],
                     model.keep_prob_in: 1.0,
                     model.keep_prob_hidden: 1.0}

        result = sess.run(ops, feed_dict=feed_dict, options=run_options)

        if args.multi:
            predicted_class = np.argmax(result[1][:, :, 1], axis=0)
            test_accs.append(sum(predicted_class == labels_query) / (nway * kquery))
        else:
            test_accs.append(result[0])

        if iteration % 100==0:
            print('Accuracy:', np.mean(test_accs))

    print('Mean test accuracy:', np.mean(test_accs))
    stds = np.std(np.array(test_accs))
    ci95 = 1.96 * stds * 100 / np.sqrt(num_batches)
    print('stds:', stds)
    print('ci95:', ci95)


def main():
    training = not args.test

    with open(os.path.join(data_path, pkl_file), 'rb') as f:
        all_files = pickle.load(f)
        print('load episodes from file, len:', len(all_files))

    model = MOE(nway, kshot, kquery, n_models, feature_size, meta_batchsz)

    # TODO: tf.summary.merge_all() returns None. Check!
    # model.summ_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # tf.global_variables() to save moving_mean and moving variance of batch norm
    # tf.trainable_variables()  NOT include moving_mean and moving_variance.

    # initialize, under interative session
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
    if os.path.exists(ckpt_name):
        # alway load ckpt both train and test.
        model_file = tf.train.latest_checkpoint(ckpt_name)
        print("Restoring model weights from ", model_file)
        saver.restore(sess, model_file)

    if args.multi:
        test(model, all_files, sess)
    else:
        if training:
            train(model, all_files, saver, sess)
        else:
            test(model, all_files, sess)


if __name__=="__main__":
    main()
