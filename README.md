# Meta-Meta Classification for One-Shot Learning: [paper](https://arxiv.org/pdf/2004.08083.pdf)

## Download the data

Execute the following script to download the data.

```
python ./utils/get_dataset_script/get_<dataset>.py
```

`<dataset>` can be `ILSVRC`, `aircraft`, `bird` and `omniglot`.

**Note:** Register at [**ImageNet**](http://www.image-net.org/) and request for a username and an access key to download ILSRVC-2012 data set. Pass them as arguments to above command. 

```
python ./utils/get_dataset_script/get_ILSVRC.py accesskey username
```

## ResNet Pretraining

1. Train ResNet-152 on ILSVRC training classes:

```
python resnet/resnet_train_imagenet.py data/ILSVRC 
```

2. Extract features for all datasets: train, test

```
python resnet_feature_extract.py  data/<dataset> --resume model_best_imagenet.pth.tar -f train
```
```
python resnet_feature_extract.py  data/<dataset> --resume model_best_imagenet.pth.tar -f test
```

Output: `.npy` files containing features of train images in `data/<dataset>/features_train` and test images in `data/<dataset>/features_test`.

## Clustering 

Cluster the training data using k-means:

```
python ./utils/cluster_images.py data/<dataset> -ds <dataset> -n num_clusters
```

Predict the nearest cluster for the test data using the learnt k-means:

```
python ./utils/cluster_images.py data/<dataset> -ds <dataset> -n num_clusters -t
```

## Run MMC for One-vs-All one-shot

1. Run MAML on individual clusters (assume dataset=ILSVRC, num_clusters=4):

```
python main.py data/ILSVRC/ ckptILSVRC_4_0/ -p filelistILSVRC_4_0 -cf cluster_4_0
python main.py data/ILSVRC/ ckptILSVRC_4_1/ -p filelistILSVRC_4_1 -cf cluster_4_1
python main.py data/ILSVRC/ ckptILSVRC_4_2/ -p filelistILSVRC_4_2 -cf cluster_4_2
python main.py data/ILSVRC/ ckptILSVRC_4_3/ -p filelistILSVRC_4_3 -cf cluster_4_3
```

Output: This creates 4 saved models for 4 clusters.

2. Create training and testing problems for meta-aggregation:

Training
```
python ./data_generators/data_generator.py
```
Testing
```
python ./data_generators/data_generator.py -t
```

Output: `filelistILSVRC` (for training) and `filelistILSVRC_test` (for testing)

3. Get the query logits (preds) for the training problems as well as test problems for the 4 individual learners:

**Training problems:**
```
python ./main_query.py data/ILSVRC/ ckptILSVRC_4_0/ -p filelistILSVRC -cf cluster_4_0 -cl 4 -m 0
python ./main_query.py data/ILSVRC/ ckptILSVRC_4_1/ -p filelistILSVRC -cf cluster_4_1 -cl 4 -m 1
python ./main_query.py data/ILSVRC/ ckptILSVRC_4_2/ -p filelistILSVRC -cf cluster_4_2 -cl 4 -m 2
python ./main_query.py data/ILSVRC/ ckptILSVRC_4_3/ -p filelistILSVRC -cf cluster_4_3 -cl 4 -m 3
```

**Test problems:**
```
python ./main_query.py data/ILSVRC/ ckptILSVRC_4_0/ -p filelistILSVRC_test -cf cluster_4_0 -cl 4 -m 0 -t
python ./main_query.py data/ILSVRC/ ckptILSVRC_4_1/ -p filelistILSVRC_test -cf cluster_4_1 -cl 4 -m 1 -t
python ./main_query.py data/ILSVRC/ ckptILSVRC_4_2/ -p filelistILSVRC_test -cf cluster_4_2 -cl 4 -m 2 -t
python ./main_query.py data/ILSVRC/ ckptILSVRC_4_3/ -p filelistILSVRC_test -cf cluster_4_3 -cl 4 -m 3 -t
```

This generates `.npy` files storing the query logits (preds) and query accuracy for all these problems.

4. Train meta-aggregator using the query preds obtained from (3) for the training problems:

```
python ./main_MC.py data/ILSVRC/ ckptILSVRC_moe/ -p filelistILSVRC -n 4
```

Output: meta-aggregator model `ckptILSVRC_moe`

5. Test meta-aggregator using the query preds obtained from (3) for the test problems:

```
python ./main_MC.py  data/ILSVRC/  ckptILSVRC_moe/ -p filelistILSVRC_test  -n 4   -t
```

Output: final accuracy, CI95


NOTE: use `-ds omniglot` in step 1, 3 for `omniglot` data-set.


## Run MMC for 5-way one-shot

1. Run MAML on individual clusters (assume num_clusters=4):

```
python main.py  data/ILSVRC/ ckptILSVRC5way_4_0/ -p filelistILSVRC5way_4_0 -cf cluster_4_0 \
     --kquery 15  --nway 5 --multi

python main.py  data/ILSVRC/ ckptILSVRC5way_4_1/ -p filelistILSVRC5way_4_1 -cf cluster_4_1 \
     --kquery 15  --nway 5 --multi

python main.py  data/ILSVRC/ ckptILSVRC5way_4_2/ -p filelistILSVRC5way_4_2 -cf cluster_4_2 \
     --kquery 15  --nway 5 --multi

python main.py  data/ILSVRC/ ckptILSVRC5way_4_3/ -p filelistILSVRC5way_4_3 -cf cluster_4_3 \
     --kquery 15  --nway 5 --multi

```


2. Create test problems for meta-aggregation:

```
python data_generators/data_generator.py -t --kquery 15 --nway 5 -p filelistILSVRC5way \
     --multi --test_problems 600 ILSVRC
```

Output: `filelistILSVRC5way_test` (for testing)

3. Get the query logits (preds) for the test problems for the 4 individual learners:

```
python main_query.py data/ILSVRC/ ckptILSVRC5way_4_0/ -p filelistILSVRC5way_test -cf cluster_4_0 \
     -cl 4 -m 0 --kquery 15 --nway 5 --multi --test_problems 600 -t
     
python main_query.py data/ILSVRC/ ckptILSVRC5way_4_1/ -p filelistILSVRC5way_test -cf cluster_4_1 \
     -cl 4 -m 0 --kquery 15 --nway 5 --multi --test_problems 600 -t

python main_query.py data/ILSVRC/ ckptILSVRC5way_4_2/ -p filelistILSVRC5way_test -cf cluster_4_2 \
     -cl 4 -m 0 --kquery 15 --nway 5 --multi --test_problems 600 -t     

python main_query.py data/ILSVRC/ ckptILSVRC5way_4_3/ -p filelistILSVRC5way_test -cf cluster_4_3 \
     -cl 4 -m 0 --kquery 15 --nway 5 --multi --test_problems 600 -t
```


4. Test meta-aggregator using the query logits (preds) obtained from (3) for the test problems:

```
python main_MC.py data/ILSVRC/ ckptILSVRC_moe/ -p filelistILSVRC5way_test \
      -n 4 --kquery 15 --nway 5 --multi -t 
```

Output: Final accuracy, CI95

## Selected arguments

- data\_path: path to the folder containing train and test images: `data/<dataset>`
- ckpt\_name: specify the path to a checkpoint to save the model and if exists, load all the parameters
- model
   - --data\_source: type of images used (omniglot or others)
   - --test: flag indicating meta-testing or working on test images
   - --multi: flag indicating multi-class problems
   - --cuda\_id: GPU to be used
- Hyperparameters
   - --train_lr: learning rate for the adapted models (inner loop): `0.0001` for clusters, `0.001` on whole data
   - --meta\_lr: learning rate for the global update of MAML (outer loop): `0.001`
   - --pkl\_file: name of the pickle file containing the list of image files in all sampled problems
   - --cluster\_folder: folder containing a particular cluster: `cluster_<num_clusters>_<cluster_id>` for clusters, `None` for whole data
   - --n_models: number of learners used: `num_clusters`
   - --kshot: number of images from each class in support set
   - --kquery: number of images from each class in query set
   - --nway: number of classes per task
   - --metabatch: number of tasks per batch: `4`
   - --steps: number of gradient update steps in the inner loop: `5`
   - --iter: number of training iterations: `40,000`
   - --train\_problems: number of tasks used for training: `100,000`
   - --test\_probolems: number of tasks used for testing: `10,000`

