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
