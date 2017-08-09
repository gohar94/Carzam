# TODO Add shebang.
"""
Creates all the same folders in test or valid folders as those in train folder.
Needed for training and testing models because keras requires same classes (directories) to be present in training, test and validation stages.
"""
import os

if __name__ == '__main__':
    mode = 'test'
    rootdir = '/home/ubuntu/deep_learning_source/data/compcars/data/image/train'
    i = 0
    for subdir, dirs, files in os.walk(rootdir):
        for directory in dirs:
            if not os.path.exists(mode + '/' + directory):
                os.makedirs(mode + '/' + directory) 
                print directory
            else:
                i += 1
        break
    print i
