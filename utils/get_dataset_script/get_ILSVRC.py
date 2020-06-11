import subprocess
import sys

cmds = []
cmds.append(['wget', 'https://github.com/tensorflow/models/blob/master/research/inception/inception/data/download_imagenet.sh'])
cmds.append(['export', 'IMAGENET_ACCESS_KEY=', sys.argv[1]])
cmds.append(['export', 'IMAGENET_USERNAME=', sys.argv[2]])
cmds.append(['wget', 'https://github.com/awslabs/deeplearning-benchmark/blob/master/tensorflow/inception/inception/data/imagenet_2012_validation_synset_labels.txt'])
cmds.append(['mv', 'imagenet_2012_validation_synset_labels.txt', 'synsets.txt'])
cmds.append(['nohup', 'bash', 'download_imagenet.sh', '.', 'synsets.txt', '>& download.log &'])
cmds.append(['wget', 'https://github.com/juliensimon/aws/blob/master/mxnet/imagenet/build_validation_tree.sh'])
cmds.append(['bash', 'build_validation_tree.sh'])
cmds.append(['python', 'utils/get_dataset_script/proc_ILSVRC.py'])

for cmd in cmds:
    print(' '.join(cmd))
    subprocess.call(cmd)