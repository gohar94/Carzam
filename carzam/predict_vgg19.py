# TODO Add shebang
"""

"""

import utils

IMG_ROWS = 224
IMG_COLS = 224
CHANNEL = 3
BATCH_SIZE = 64
DATA_PATH = "../../data/compcars/data/image/"

def vgg19_model(img_rows, img_cols, channel=1, num_classes=None):
    """
    VGG 19 Model for Keras

    Model Schema is based on
    https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d

    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilZ2RVeVhKY0FyRmc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(channel, img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('imagenet_models/vgg19_weights.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model

def main():
    batches = utils.get_batches(DATA_PATH+'train', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=BATCH_SIZE)
    val_batches = utils.get_batches(DATA_PATH+'valid', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=BATCH_SIZE)
    test_batches = utils.get_batches(DATA_PATH+'test', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), shuffle=False, batch_size=BATCH_SIZE, class_mode=None)
    model_vgg19 = vgg19_model(img_rows, img_cols, channel, batches.nb_class)
    model_vgg19.load_weights('vgg19_model_60.h5')
    probs_vgg19 = model_vgg19.predict_generator(test_batches, test_batches.nb_sample)
    labels_vgg19 = test_batches.classes
    labels_predicted_vgg19 = [np.argmax(prob) for prob in probs_vgg19]
    classes_vgg19 = [classes_ids[idx] for idx in labels_predicted_vgg19]
    correct_vgg19, incorrect_vgg19 = check_count_correct(filenames, classes_vgg19)

    print "Probabilities"
    print probs_vgg19
    print "Labels"
    print labels_vgg19
    print "Top 1: Correct %d, Incorrect %d" % (correct_vgg19, incorrect_vgg19)

    top_5_labels_pred_vgg19 = [np.argpartition(prob, -5)[-5:] for prob in probs_vgg19] # get indices for top 5 probs
    print top_5_labels_pred_vgg19
    classes_top_5_vgg19 = []
    for i in range(len(top_5_labels_pred_vgg19)):
        classes_temp = [classes_ids[idx] for idx in top_5_labels_pred_vgg19[i]]
        classes_top_5_vgg19.append(classes_temp)
    print classes_top_5_vgg19
    correct_vgg19_top5, incorrect_vgg19_top5 = check_count_correct_top_5(filenames, classes_top_5_vgg19)
    print "Top 5: Correct %d, Incorrect %d" % (correct_vgg19_top5, incorrect_vgg19_top5)

if __name__ == '__main__':
    main()
