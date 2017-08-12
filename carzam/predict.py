# TODO Add shebang
"""

"""
from keras.preprocessing import image
import numpy as np
import utils
import pickle
from compcars_vgg19_model import vgg19_model
from compcars_inception_v1_model import googlenet_model

class PredictionResults(object):
    """
    # TODO Fill docstring.
    """
    def __init__(self):
        """
        # TODO Fill docstring.
        """
        probabilities = None
        classes_ids = None
        filenames = None

def predict(model_name, weights_filename):
    """
    Predicts the test images and shows how many correct and incorrect predictions were made for Top 1 and Top 5 classes.

    @param model_name: String containing the name of the model to use for prediction.
    @param weights_filename: String containing the filename of the weights file to load for the corresponding model.

    @return: PredictionResults object if successful, else None.
    """
    img_rows = 224
    img_cols = 224
    channel = 3
    batch_size = 64
    data_path = "../data/compcars/data/image/"
    model_path = "../models/"
    imagenet_model_path = "../imagenet_models/"

    # Get images
    batches = utils.get_batches(data_path+'train', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=batch_size, shuffle=False, class_mode=None)
    test_batches = utils.get_batches(data_path+'test', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), shuffle=False, batch_size=batch_size, class_mode=None)
    
    # Create model and load weights
    print "Using %s model" % model_name
    if model_name == "vgg19":
        model = vgg19_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    elif model_name == "inception_v1":
        model = googlenet_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    else:
        return None
    model.load_weights(model_path + weights_filename)
    
    # Predict
    probs = model.predict_generator(test_batches, test_batches.nb_sample)
    labels = test_batches.classes
    filenames = test_batches.filenames

    # Get a list of all the class labels
    classes_ids = {v: k for k, v in batches.class_indices.iteritems()}
    
    # Process the results for Top 1
    print "First filename is " 
    print filenames[0]
    print "Probs for it are"
    print probs[0]
    labels_predicted = [np.argmax(prob) for prob in probs]
    classes = [classes_ids[idx] for idx in labels_predicted]
    correct, incorrect = utils.count_correct_compcars(filenames, classes)

    # Process the results for Top 5
    top_5_labels_pred = [np.argpartition(prob, -5)[-5:] for prob in probs]
    classes_top_5 = []
    for i in range(len(top_5_labels_pred)):
        classes_temp = [classes_ids[idx] for idx in top_5_labels_pred[i]]
        classes_top_5.append(classes_temp)
    correct_top_5, incorrect_top_5 = utils.count_correct_compcars_top_k(filenames, classes_top_5)
    
    print "Top 1: Correct %d, Incorrect %d" % (correct, incorrect)
    print "Top 5: Correct %d, Incorrect %d" % (correct_top_5, incorrect_top_5)

    results = PredictionResults()
    results.probabilities = probs
    results.filenames = filenames
    results.classes_ids = classes_ids

    return results
    """
    classes_ids = {v: k for k, v in batches.class_indices.iteritems()}
    classes_inds = model.predict_classes(test_batches, batch_size)
    classes = [classes_ids[idx] for idx in classes_inds]
    filenames = test_batches.filenames
    correct, incorrect = utils.count_correct_compcars(filenames, classes)
    """

def combined_prediction(results_a, results_b):
    """
    Reports the correct and incorrect number of predictions in two PredictionResults objects for Top 1 and Top 5 classes.
    
    @param results_a: First PredictionResults object.
    @param results_b: Second PredictionResults object.
    """
    assert(results_a.classes_ids == results_b.classes_ids)
    assert(results_a.filenames == results_b.filenames)
    
    classes_ids = results_a.classes_ids
    filenames = results_a.filenames
    avg_probs = utils.average_probabilities(results_a.probabilities, results_b.probabilities)
    labels_predicted_combined = [np.argmax(prob) for prob in avg_probs]
    classes_combined = [classes_ids[idx] for idx in labels_predicted_combined]
    correct, incorrect = utils.count_correct_compcars(filenames, classes_combined) 
    top_5_labels_pred_combined = [np.argpartition(prob, -5)[-5:] for prob in avg_probs]
    classes_top_5_combined = []
    for i in range(len(top_5_labels_pred_combined)):
        classes_temp = [classes_ids[idx] for idx in top_5_labels_pred_combined[i]]
        classes_top_5_combined.append(classes_temp)
    correct_top_5, incorrect_top_5 = utils.count_correct_compcars_top_k(filenames, classes_top_5_combined)

    print "Combined prediction results:"
    print "Top 1: Correct %d, Incorrect %d" % (correct, incorrect)
    print "Top 5: Correct %d, Incorrect %d" % (correct_top_5, incorrect_top_5)

def main():
    results_vgg19 = predict("vgg19", "vgg19_model_60.h5")
    pickle.dump(results_vgg19, open("results_vgg19.p", "wb"))
    # results_vgg19 = pickle.load(open("results_vgg19.p", "rb"))
    # results_inception_v1 = predict("inception_v1", "inception_model_adam_100.h5")
    # pickle.dump(results_inception_v1, open("results_inception_v1.p", "wb"))
    # results_inception_v1 = pickle.load(open("results_inception_v1.p", "rb"))
    # combined_prediction(results_vgg19, results_inception_v1)

if __name__ == '__main__':
    main()
