# TODO Add shebang
"""

"""
from keras.preprocessing import image
import cv2
import numpy as np
import utils
import pickle
import sys
from compcars_vgg19_model import vgg19_model
from compcars_inception_v1_model import googlenet_model

def predict(image_file_path):
    """

    """
    img = cv2.imread(image_file_path)

    img_rows = 224
    img_cols = 224
    channel = 3
    batch_size = 64
    data_path = "../data/compcars/data/image/"
    model_path = "../models/"
    imagenet_model_path = "../imagenet_models/"

    batches = utils.get_batches(data_path+'train', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=batch_size)

    model_vgg19 = vgg19_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    model_inception_v1 = googlenet_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)

    model_vgg19.load_weights(model_path + 'vgg19_model_60.h5')
    model_inception_v1.load_weights(model_path + 'inception_model_adam_100.h5')

    probs = model_vgg19.predict(img)
    print probs

if __name__ == '__main__':
    predict(sys.argv[1])
