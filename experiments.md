# Download the data

Execute the following script to download the data.
```python ./utils/get_dataset_script/get_<dataset>.py```

`<dataset>` can be `ILSVRC`, `aircraft`, `bird` and `omniglot`.

# Clustering 

Cluster on training data:

```
python ./utils/cluster_images.py    data/<dataset>    -ds <dataset>    -n num_clusters
```

Cluster on test data using the same model:

```
python ./utils/cluster_images.py    data/<dataset>   -ds <dataset>    -n num_clusters    -t
```



# Run MMC

1. Run MAML on individual clusters (assume num_clusters=4):

```
python main.py    data/ILSVRC/     ckptILSVRC_4_0/    -p filelistILSVRC_4_0     -cf cluster_4_0
```

(similarly run on all clusters)

This creates 4 saved models for 4 clusters.

2. Create training and testing problems for meta-aggregation:

```
python ./data_generators/data_generator.py
```
(with both training=True and training=False at line 291)

3. This generates `filelistILSRV` (for training), `filelistILSRV_test` (for testing)

* Get the query preds for the training problems as well as test problems for all the cluster models:

**Training problems:**
```
python ./main_query.py    data/ILSVRC/    ckptILSVRC_4_0/    -p filelistILSRV    -cf cluster_4_0    -cl 4   -m 0
```

**Test problems:**
```
python ./main_query.py    data/ILSVRC/    ckptILSVRC_4_0/    -p filelistILSRV_test    -cf cluster_4_0    -cl 4   -m 0    -t
```

(similarly run on all clusters)

This generates `npy` files storing the query preds and query accuracy for all these problems.

4. Train meta-classifier using the query preds obtained above for the training problems:

```
python ./main_MC.py data/ILSVRC/  ckptILSVRC_moe/ -p filelistILSRV    -n 4
```

O/P: meta-classifier model ckptILSVRC_moe

5. Test meta-classifier using the query preds obtained above for the training problems:

```
python ./main_MC.py  data/ILSVRC/  ckptILSVRC_moe/ -p filelistILSRV  -n 4   -t
```

output: final accuracy, CI95


NOTE: use -ds omniglot in step 1, 3 for omniglot data-set.


# Run MMC for 5-way one-shot
