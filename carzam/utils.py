# TODO Add shebang
"""

"""

from keras.preprocessing import image

import numpy as np

def get_batches(path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
    """
    Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.

    See Keras documentation: https://keras.io/preprocessing/image/

    # TODO Params and return docstring.
    """
    return gen.flow_from_directory(path, target_size=(224,224), class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

def vgg_preprocess(x):
    """
    Subtracts the mean RGB value, and transposes RGB to BGR.
    The mean RGB was computed on the image set used to train the VGG model.

    @param x: Image array (height x width x channels)
    
    @return: Image array (height x width x transposed_channels)
    """
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

def count_correct_compcars(filenames, classes):
    """
    Takes the filenames and predicted classes and counts the correct and incorrect predictions.
    Each index of filenames and classes corresponds to the same test image.

    @param filenames: List of filenames of the test images in CompCars format.
    @param classes: List of predicted classes of the images.

    @return correct: Number of correct predictions.
    @return incorrect: Number of incorrect predictions.
    """
    assert(len(filenames) == len(classes))
    correct = incorrect = 0
    for i in range(len(filenames)):
        if filenames[i].split('/')[0] == classes[i]:
            correct += 1
        else:
            incorrect += 1
    return correct, incorrect

def count_correct_compcars_top_k(filenames, classes_top_k):
    """
    Takes the filenames and a list of top k classes according to the probability and counts the correct and incorrect predictions.
    Each index of filenames and classes corresponds to the same test image.

    @param filenames: List of filenames of the test images in CompCars format.
    @param classes: List of lists of top k predicted classes of the images.

    @return correct: Number of correct predictions.
    @return incorrect: Number of incorrect predictions.
    """
    assert(len(filenames) == len(classes_top_k))
    correct = incorrect = 0
    for i in range(len(filenames)):
        if filenames[i].split('/')[0] in classes_top_k[i]:
            correct += 1
        else:
            incorrect += 1
    return correct, incorrect

def average_probabilities(probs_a, probs_b):
    """
    Takes the elementwise mean of the numbers in both lists and returns a list of the same length as the input lists.

    @param probs_a: First list of elements to average.
    @param probs_b: Second list of elements to average.

    @return: List of averages elements.
    """
    assert(len(probs_a) == len(probs_b))
    final = []
    for i in range(len(probs_a)):
        temp = []
        for j in range(len(probs_a[i])):
            temp.append((probs_a[i][j]+probs_b[i][j])/2)
        final.append(temp)
    return final
