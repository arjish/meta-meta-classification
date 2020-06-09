import subprocess


cmds = []
cmds.append(['wget', 'wget https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip'])
cmds.append(['wget', 'https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip'])
cmds.append(['unzip', 'images_background.zip'])
cmds.append(['unzip', 'images_evaluation.zip'])
cmds.append(['mkdir', 'omniglot'])
cmds.append(['mv', 'images_background/*', 'omniglot'])
cmds.append(['mv', 'images_evaluation/*', 'omniglot'])
cmds.append(['python', 'utils/get_dataset_script/proc_omniglot.py'])
cmds.append(['rm', '-rf', 'images_background.zip', 'images_background', 'images_evaluation.zip', 'images_evaluation'])

for cmd in cmds:
    print(' '.join(cmd))
    subprocess.call(cmd)
