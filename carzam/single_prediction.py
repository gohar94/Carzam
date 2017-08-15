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

def class_id_to_strings(class_id, make_strings, model_strings):
    """
    TODO Docstring.
    """
    make_id = class_id.split('_')[0]
    model_id = class_id.split('_')[1]
    try:
        # Error checking to avoid out of bounds errors
        return make_strings[int(make_id)-1], model_strings[int(model_id)-1]
    except:
        return None, None

def predict_and_get_probabilities(model, batches, img):
    """

    """
    probs = model.predict(img)
    assert(len(probs) == 1)
    get_prediction_classes(probs, batches)

    return probs

def get_prediction_classes(probs, batches):
    """

    """
    classes_ids = {v: k for k, v in batches.class_indices.iteritems()}
    labels_predicted = [np.argmax(prob) for prob in probs]
    classes = [classes_ids[idx] for idx in labels_predicted]
    top_5_labels_pred = np.argpartition(probs[0], -5)[-5:]
    classes_top_5 = [classes_ids[idx] for idx in top_5_labels_pred]
    print classes
    print classes_top_5

    return classes, classes_top_5

def get_classes_to_output(classes, classes_top_5, make_strings, model_strings):
    """
    TODO Docstring.
    """
    classes_out = []
    classes_out_top_5 = []
    for class_id in classes:
        make_string, model_string = class_id_to_strings(class_id, make_strings, model_strings)
        classes_out.append({'class_id': class_id, 'make': make_string, 'model': model_string})
    for class_id in classes_top_5:
        make_string, model_string = class_id_to_strings(class_id, make_strings, model_strings)
        classes_out_top_5.append({'class_id': class_id, 'make': make_string, 'model': model_string})
    return classes_out, classes_out_top_5

def get_loaded_models_and_batches(img_rows, img_cols, channel, batch_size, data_path, model_path, imagenet_model_path):
    """
    # TODO Docstring.
    """
    # Get training set image batches. TODO Might want to pickle these and not load every time.
    batches = utils.get_batches(data_path+'train', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=batch_size, shuffle=False, class_mode=None)

    # Construct models.
    model_vgg19 = vgg19_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    model_inception_v1 = googlenet_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)

    # Load weights.
    model_vgg19.load_weights(model_path + 'vgg19_model_60.h5')
    model_inception_v1.load_weights(model_path + 'inception_model_adam_100.h5')

    return batches, model_vgg19, model_inception_v1

def get_image(img_rows, img_cols, image_file_path):
    """
    TODO Doctring.
    """
    # DO NOT CHANGE THE ORDER OF THE NEXT 4 LINES - JUST. DO. NOT.
    img = image.load_img(image_file_path, target_size=(img_rows, img_cols))
    img = image.img_to_array(img)
    img = utils.vgg_preprocess(img)
    img = np.expand_dims(img, axis=0)

    return img

def perform_combined_prediction(model_vgg19, model_inception_v1, batches, img):
    """
    TODO Docstring.
    """
    probs_inception_v1 = predict_and_get_probabilities(model_inception_v1, batches, img)
    probs_vgg19 = predict_and_get_probabilities(model_vgg19, batches, img)
    avg_probs = utils.average_probabilities(probs_inception_v1, probs_vgg19)
    classes, classes_top_5 = get_prediction_classes(avg_probs, batches)

    return classes, classes_top_5

def run(image_file_path):
    """
    # TODO Docstring.
    """
    img_rows = 224
    img_cols = 224
    channel = 3
    batch_size = 64
    data_path = "../data/compcars/data/image/"
    model_path = "../models/"
    imagenet_model_path = "../imagenet_models/"

    batches, model_vgg19, model_inception_v1 = get_loaded_models_and_batches(img_rows, img_cols, channel, batch_size, data_path, model_path, imagenet_model_path)
    img = get_image(img_rows, img_cols, image_file_path)
    classes, classes_top_5 = perform_combined_prediction(model_vgg19, model_inception_v1, batches, img)

    return classes, classes_top_5

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Incomplete arguments"
    else:
        run(sys.argv[1])
