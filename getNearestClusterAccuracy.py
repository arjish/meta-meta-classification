import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_path', metavar='DATA',
    help='path to data')
parser.add_argument('-n', '--n_clusters', default=16, type=int,
    help='# of models (default=16)')
parser.add_argument('-p', '--pkl_file', default='filelist',  type=str,
    help='path to pickle file')

args = parser.parse_args()

data_path = args.data_path
n_clusters = args.n_clusters
pkl_file = args.pkl_file

clusterLabelsFile = 'clusterLabels_'+str(n_clusters)+'_'+pkl_file + '.npy'
labels = np.load(os.path.join(data_path, 'CLUSTER_'+str(n_clusters), clusterLabelsFile))
print('Shape of labels:', labels.shape)
acc_list = []
for i in range(n_clusters):
    acc_file = os.path.join(data_path, 'CLUSTER_'+str(n_clusters),
        'queryAcc_' + pkl_file + '_cluster' + str(n_clusters) + '_' + str(i) + '.npy')
    acc_list.append(np.load(acc_file))

acc_list = np.stack(acc_list, axis=1)
print('Shape of acc_list:', acc_list.shape)

accs_nearest = acc_list[np.arange(len(labels)), labels]

mean_accs = np.mean(accs_nearest)

print('Mean accuracy using nearest model:', mean_accs)
stds = np.std(accs_nearest)
ci95 = 1.96 * stds * 100 / np.sqrt(accs_nearest.shape[0])
print('stds:', stds)
print('ci95:', ci95)
