import os
import shutil

if __name__ == '__main__':
    path = '/home/ubuntu/deep_learning_source/data/compcars/data/image/train'
    for subdirs, dirs, files in os.walk(path):
        for directory in dirs:
            print directory
            i = 0
            for filename in os.listdir('train/'+directory):
		print filename
                if not os.path.exists('valid/'+directory):
                    os.makedirs('valid/'+directory)
                shutil.move('train/'+directory+'/'+filename, 'valid/'+directory+'/'+filename)
                i += 1
                if i == 5: # max num from each class to put in validation set
                    break
