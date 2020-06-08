import os
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('data_path', metavar='DATA',
    help='path to data')
parser.add_argument('-s', '--soft', action='store_true', default=False,
    help='set for soft bagging, otherwise hard bagging')
parser.add_argument('-n', '--n_models', default=16, type=int,
    help='# of models (default=16)')
parser.add_argument('-p', '--pkl_file', default='filelist',  type=str,
    help='path to pickle file')
parser.add_argument('-nd', '--n_data_points', default=10000, type=int,
    help='# of problems (default=10,000)')
parser.add_argument('--nway', default=51, type=int,
    help='# of classes per problem (default=51)')
parser.add_argument('--kquery', default=1, type=int,
    help='# of queries per class (default=1)')

args = parser.parse_args()

n_models = args.n_models
n_data_points = args.n_data_points
pkl_file = args.pkl_file
data_path = args.data_path
nway = args.nway
kquery = args.kquery 


def myEntropy(count_labels, base=None):
    """ Computes entropy of label distribution. """
    count_labels = count_labels[count_labels!=0]
    tot_labels = sum(count_labels)
    if tot_labels <= 1:
        return 0
    probs = count_labels / tot_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent

def test():
    threshold = math.ceil(n_models / 2)

    labels_query = [0] * ((nway - 1) * kquery)
    labels_query.extend([1] * (nway - 1) * kquery)

    query_preds_list = []
    for i in range(n_models):
        preds_file = os.path.join(data_path, 'WHOLE', 'queryPreds_' + pkl_file + '_model' + str(i) + '.npy')
        query_preds_list.append(np.load(preds_file))

    if args.soft:
        query_preds= np.mean(np.array(query_preds_list), axis=0)
        query_predictions = np.argmax(query_preds, axis=-1)
        # entropy_probs = 0
        # for i in range(n_data_points):
        #     for j in range(2*kquery):
        #         entropy_probs += entropy(query_probs[i][j])
        # print('Average ensemble entropy:', entropy_probs/ (n_data_points * 2 * (nway - 1) * kquery))
    else:
        query_preds_list = np.argmax(np.array(query_preds_list), axis=-1)
        query_votes = np.count_nonzero(query_preds_list, axis=0)
        query_predictions = (query_votes >= threshold)

    correct_predictions = np.sum(query_predictions == labels_query, axis=1)
    test_accs = correct_predictions / (2 * (nway - 1) * kquery)

    print('Mean test accuracy:', np.mean(test_accs))
    stds = np.std(test_accs)
    ci95 = 1.96 * stds * 100 / np.sqrt(n_data_points)
    print('stds:', stds)
    print('ci95:', ci95)


def main():
    test()

if __name__=="__main__":
    main()
