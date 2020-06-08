import os
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true', default=False, help='set for test, otherwise train')
parser.add_argument('-p', '--pkl_file', default="filelist900_0",  type=str,  metavar='PKL', help='path to pickle')
parser.add_argument('-d', '--data_folder', default="imagenet_1K_900", type=str, metavar='DATA', help='path to data')
parser.add_argument('-n', '--n_clusters', default=16, type=int, metavar='CLUSTERS', help='num of clusters (default=15)')

args = parser.parse_args()

mode = 'test' if args.test else 'train'
pkl_file = args.pkl_file
data_folder = args.data_folder
n_clusters = args.n_clusters

def main():
    kshot = 1
    kquery = 50  # 50
    target_interval = kshot + 3 * kquery

    if os.path.exists('target_'+pkl_file):
        with open('target_'+pkl_file, 'rb') as f:
            filenames = pickle.load(f)
            print('load episodes from target file, len:', len(filenames))
    else:
        with open(pkl_file, 'rb') as f:
            filenames = pickle.load(f)
            print('load episodes from pkl file, len:', len(filenames))
        filenames = [filenames[i] for i in
                            range(kquery, len(filenames), target_interval)]
        with open('target_'+pkl_file, 'wb') as f:
            pickle.dump(filenames, f)
            print('load episodes from target file, len:', len(filenames))


    labels = []
    for i, file in enumerate(filenames):
        for cl in range(n_clusters):
            replacedBy = os.path.join('CLUSTER_'+str(n_clusters),
                'cluster_'+str(n_clusters)+'_'+str(cl), mode)
            #print(replacedBy)
            target_path = file.replace(mode, replacedBy)
            #print(target_path)
            if os.path.exists(target_path):
                labels.append(cl)
                break
            elif cl == n_clusters-1:
                print('Not found:\n', file)

    clusterLabelsFile = 'clusterLabels_'+str(n_clusters)+'_'+pkl_file
    np.save(os.path.join(data_folder, 'CLUSTER_'+str(n_clusters), clusterLabelsFile), labels)


if __name__ == "__main__":
    main()
