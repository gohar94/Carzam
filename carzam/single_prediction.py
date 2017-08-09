# TODO Add shebang
"""

"""
from keras.preprocessing import image
from PIL import Image
import numpy as np
import utils
import pickle
import sys
from compcars_vgg19_model import vgg19_model
from compcars_inception_v1_model import googlenet_model

def predict(image_file_path):
    """

    """
    img_rows = 224
    img_cols = 224
    channel = 3
    batch_size = 64
    data_path = "../data/compcars/data/image/"
    model_path = "../models/"
    imagenet_model_path = "../imagenet_models/"

    img = Image.open(image_file_path)
    if img.size != (img_rows, img_cols):
        img = img.resize((img_rows, img_cols))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.vgg_preprocess(x)

    batches = utils.get_batches(data_path+'train', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=batch_size)

    model_vgg19 = vgg19_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    model_inception_v1 = googlenet_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)

    model_vgg19.load_weights(model_path + 'vgg19_model_60.h5')
    model_inception_v1.load_weights(model_path + 'inception_model_adam_100.h5')

    probs = model_vgg19.predict(x)
    print probs
    
    # Get a list of all the class labels
    classes_ids = list(iter(batches.class_indices))
    for c in batches.class_indices:
        classes_ids[batches.class_indices[c]] = c
    labels_predicted = [np.argmax(prob) for prob in probs]
    classes = [classes_ids[idx] for idx in labels_predicted]
    top_5_labels_pred = [np.argpartition(prob, -5)[-5:] for prob in probs]
    classes_top_5 = []
    for i in range(len(top_5_labels_pred)):
        classes_temp = [classes_ids[idx] for idx in top_5_labels_pred[i]]
        classes_top_5.append(classes_temp)
    print classes
    print classes_top_5

if __name__ == '__main__':
    predict(sys.argv[1])
