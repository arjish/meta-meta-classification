# Download the data

Execute the following script to download the data.

```
python ./utils/get_dataset_script/get_<dataset>.py
```

`<dataset>` can be `ILSVRC`, `aircraft`, `bird` and `omniglot`.

TODO: Add section for imagenet

# Clustering 

Cluster on training data:

```
python ./utils/cluster_images.py data/<dataset> -ds <dataset> -n num_clusters
```

Cluster on test data using the same model:

```
python ./utils/cluster_images.py data/<dataset> -ds <dataset> -n num_clusters -t
```

# Run MMC

1. Run MAML on individual clusters (assume num_clusters=4):

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

TODO: (with both training=True and training=False at line 291)

Output: `filelistILSVRC` (for training) and `filelistILSVRC_test` (for testing)

3. Get the query logits(preds) for the training problems as well as test problems for all the cluster models:

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

This generates `npy` files storing the query logits(preds) and query accuracy for all these problems.

4. Train meta-aggregator using the query preds obtained above for the training problems:

```
python ./main_MC.py data/ILSVRC/ ckptILSVRC_moe/ -p filelistILSVRC -n 4
```

Output: meta-aggregator model `ckptILSVRC_moe`

5. Test meta-aggregator using the query preds obtained above for the training problems:

```
python ./main_MC.py  data/ILSVRC/  ckptILSVRC_moe/ -p filelistILSVRC_test  -n 4   -t
```

Output: final accuracy, CI95


NOTE: use `-ds omniglot` in step 1, 3 for `omniglot` data-set.


# Run MMC for 5-way one-shot

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


2. Create training and testing problems for meta-aggregation:

```
python data_generators/data_generator.py -t --kquery 15 --nway 5 -p filelistILSVRC5way --multi --test_problems 600 ILSVRC
```

Output: `filelistILSVRC5way_test` (for testing)

3. Get the query logits(preds) for the test problems for all the cluster models:

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


4. Test meta-classifier using the query logits(preds) obtained above for the training problems:

```
python main_MC.py data/ILSVRC/ ckptILSVRC_moe/ -p filelistILSVRC5way_test \
      -n 4 --kquery 15 --nway 5 --multi -t 
```

Output: Final accuracy, CI95
