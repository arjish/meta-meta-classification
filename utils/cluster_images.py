import numpy as np
from sklearn.cluster import KMeans
import pickle
import os, shutil
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('data_path', metavar='DATA',
    help='path to data (which is clustered)')
parser.add_argument('-t', '--test', action='store_true', default=False,
    help='set to use existing kmeans model, otherwise do clustering')
parser.add_argument('-ds', '--data_source', default='CUB', type=str,
    help='data_source (whose clustering model is used)')
parser.add_argument('-n', '--num_clusters', default=16, type=int,
    help='Number of models (default=16)')

args = parser.parse_args()


data_path = args.data_path
data_source = args.data_source
num_clusters = args.num_clusters

cluster_path_prefix = os.path.join(data_path, 'cluster_' + str(num_clusters))

def createFolderStructure(folder_list, mode):
    for i in range(num_clusters):
        for folder_name in folder_list:
            if not os.path.exists(os.path.join(cluster_path_prefix + '_' + str(i), mode, folder_name)):
                os.makedirs(os.path.join(cluster_path_prefix + '_' + str(i), mode, folder_name))


def getImageFeatures(feature_path, folder_list):
    all_image_features = []
    num_images_eachClass = []
    for folder in folder_list:
        print('\tWorking on folder', folder)
        featureFile_list = sorted(next(os.walk(os.path.join(feature_path, folder)))[2])
        all_image_features.extend([np.load(os.path.join(feature_path, folder, file))
                                           for file in featureFile_list])
        num_images_eachClass.append(len(featureFile_list))

    return np.array(all_image_features), num_images_eachClass


def copyImagesToClusters(folder_list, mode, num_images_eachClass, labels_clusters):
    startImageID = 0
    for i in range(len(folder_list)):
        print('\t Working on folder:', folder_list[i], '\t Num of images:', num_images_eachClass[i])

        folder_name = folder_list[i]
        imageFiles_list = sorted(next(os.walk(os.path.join(data_path, mode, folder_name)))[2])
        assert num_images_eachClass[i] == len(imageFiles_list), "Num of images in the folder doesn't match!."

        for im_indx in range(len(imageFiles_list)):
            clusterID = labels_clusters[startImageID+im_indx]
            shutil.copy(os.path.join(data_path, mode, folder_name, imageFiles_list[im_indx]),
                        os.path.join(cluster_path_prefix + '_' + str(clusterID),
                                     mode, folder_name))

        startImageID += num_images_eachClass[i]


kmeans_model_file = os.path.join(data_path, 'kmeans_'+data_source+'_'+str(num_clusters))
print('kmeans_model_file:', kmeans_model_file)

## Create the necessary folder tree ::
if not args.test:
    ## For training data ::
    classFoldersTrain_list = [label \
                              for label in os.listdir(os.path.join(data_path, 'train')) \
                              if os.path.isdir(os.path.join(data_path, 'train', label))]
    classFoldersTrain_list.sort()
    
    if os.path.exists(kmeans_model_file):
        with open(kmeans_model_file, 'rb') as f:
            kmeans = pickle.load(f)
        print('kmeans model loaded')
        num_images_eachClass = np.load(os.path.join(data_path, 'num_images_eachClass_train'))
    else:
        createFolderStructure(classFoldersTrain_list, 'train')

        feature_path_train = os.path.join(data_path, 'features_train')
        print('Getting the training image features....')
        all_image_features, num_images_eachClass = getImageFeatures(feature_path_train, classFoldersTrain_list)

        ## Run the clustering :
        print('\nRunning K-Means clustering...')
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_image_features)
        np.save(os.path.join(data_path, 'num_images_eachClass_train'), num_images_eachClass)

        pickle.dump(kmeans, open(kmeans_model_file, 'wb'))
        print('Model saved!')

    
    labels_clusters = kmeans.labels_
    print('Number of lables on train data:', len(labels_clusters))
    np.save(os.path.join(data_path, data_source+'_cluster_trainLabels_' + str(num_clusters)), labels_clusters)

    ## Copy the images to corresponding clusters :
    print('\nCopying training images to corresponding cluster folders...')
    copyImagesToClusters(classFoldersTrain_list, 'train', num_images_eachClass, labels_clusters)

else:
    ## For test data ::
    classFoldersTest_list = [label \
                          for label in os.listdir(os.path.join(data_path, 'test')) \
                          if os.path.isdir(os.path.join(data_path, 'test', label))]
    classFoldersTest_list.sort()

    createFolderStructure(classFoldersTest_list, 'test')

    with open(kmeans_model_file, 'rb') as f:
        kmeans = pickle.load(f)
    print('kmeans model loaded')


    feature_path_test = os.path.join(data_path, 'features_test')

    print('\nGetting the test image features....')
    all_image_features, num_images_eachClass = getImageFeatures(feature_path_test, classFoldersTest_list)
    print(all_image_features.shape)


    # Cluster label predictions :
    print('\nPredict cluster labels for test images...')
    labels_clusters = kmeans.predict(all_image_features)

    np.save(os.path.join(data_path, data_source+'_cluster_testLabels_'+str(num_clusters)), labels_clusters)
    print('Number of lables on test data:', len(labels_clusters))

    ## Copy the images to corresponding clusters :
    print('\nCopying test images to corresponding cluster folders...')
    copyImagesToClusters(classFoldersTest_list, 'test', num_images_eachClass, labels_clusters)

