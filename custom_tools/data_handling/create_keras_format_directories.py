# TODO Add shebang
"""
Scans the list for training or test data and moves the corresponding images from the main image directory to a format supported by Keras for model training, validation or testing.
"""
import os
import shutil

if __name__ == '__main__':
    mode = 'test'
    fname = '/home/ubuntu/deep_learning_source/data/compcars/data/train_test_split/classification/' + mode + '.txt'
    with open(fname) as f:
        content = f.readlines()
    for image in content:
        image = image.rstrip()
        image_split = image.split('/')
        del image_split[2] # remove the year
        directory = mode + '/' + '_'.join(image_split[:2]) + '/'
        image_name = image_split[2]
        if not os.path.exists(directory):
            os.makedirs(directory) 
        shutil.copy(image, directory+image_name)
