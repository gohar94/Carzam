# TODO Add shebang
"""

"""
from keras.preprocessing import image
import numpy as np
import utils
import pickle
import sys
from compcars_vgg19_model import vgg19_model
from compcars_inception_v1_model import googlenet_model
from keras import backend as K
K.set_image_dim_ordering('th')

def predict(model, batches, test_batches):
    # probs = model.predict(x)
    probs = model.predict_generator(test_batches, test_batches.nb_sample)
    assert(len(probs) == 1)
    classes_ids = {v: k for k, v in batches.class_indices.iteritems()}
    labels_predicted = [np.argmax(prob) for prob in probs]
    classes = [classes_ids[idx] for idx in labels_predicted]
    top_5_labels_pred = [np.argpartition(prob, -5)[-5:] for prob in probs]
    classes_top_5 = []
    for i in range(len(top_5_labels_pred)):
        classes_temp = [classes_ids[idx] for idx in top_5_labels_pred[i]]
        classes_top_5.append(classes_temp)
    print classes
    print classes_top_5

def run(image_file_path):
    """

    """
    img_rows = 224
    img_cols = 224
    channel = 3
    batch_size = 64
    data_path = "../data/compcars/data/image/"
    model_path = "../models/"
    imagenet_model_path = "../imagenet_models/"

    x = image.load_img(image_file_path, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    # x = utils.vgg_preprocess(x)

    batches = utils.get_batches(data_path+'train', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=batch_size, shuffle=False, class_mode=None)
    test_batches = utils.get_batches('../data/crap', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=1, shuffle=False, class_mode=None)

    model_vgg19 = vgg19_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    # model_inception_v1 = googlenet_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)

    model_vgg19.load_weights(model_path + 'vgg19_model_60.h5')
    # model_inception_v1.load_weights(model_path + 'inception_model_adam_100.h5')
    print model_vgg19.predict(x)

    # predict(model_inception_v1, x, batches)
    # predict(model_vgg19, batches, test_batches)
    

if __name__ == '__main__':
    run(sys.argv[1])
