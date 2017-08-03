import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import SGD, RMSprop, Adam
import vgg16 as jeremy_vgg16
from matplotlib import pyplot as plt
from googlenet import create_googlenet

"""
BATCH_SIZE = 64
DATA_PATH = "../data/compcars/data/image/"
batches = jeremy_vgg16.Vgg16().get_batches(DATA_PATH+'train',
                                 gen=image.ImageDataGenerator(preprocessing_function=jeremy_vgg16.vgg_preprocess),
                                 batch_size=BATCH_SIZE)
val_batches = jeremy_vgg16.Vgg16().get_batches(DATA_PATH+'valid',
                                     gen=image.ImageDataGenerator(preprocessing_function=jeremy_vgg16.vgg_preprocess),
                                     batch_size=BATCH_SIZE)
"""
keras_kgg = create_googlenet('googlenet_weights.h5')
keras_kgg.pop()
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# Plot the results of the training
def plot_results(history):
    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

history = keras_vgg.fit_generator(batches,
                        validation_data=val_batches,
                        samples_per_epoch=batches.nb_sample,
                        nb_val_samples=val_batches.nb_sample,
                        nb_epoch=10)
keras_vgg.save_weights('inception_model_adam_10.h5')

history = keras_vgg.fit_generator(batches,
                        validation_data=val_batches,
                        samples_per_epoch=batches.nb_sample,
                        nb_val_samples=val_batches.nb_sample,
                        nb_epoch=20)
keras_vgg.save_weights('inception_model_adam_30.h5')

history = keras_vgg.fit_generator(batches,
                        validation_data=val_batches,
                        samples_per_epoch=batches.nb_sample,
                        nb_val_samples=val_batches.nb_sample,
                        nb_epoch=30)
keras_vgg.save_weights('inception_model_adam_60.h5')

history = keras_vgg.fit_generator(batches,
                        validation_data=val_batches,
                        samples_per_epoch=batches.nb_sample,
                        nb_val_samples=val_batches.nb_sample,
                        nb_epoch=40)
keras_vgg.save_weights('inception_model_adam_100.h5')

