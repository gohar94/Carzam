import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import SGD, RMSprop, Adam
import vgg16 as jeremy_vgg16
from matplotlib import pyplot as plt

BATCH_SIZE = 64
DATA_PATH = "../data/compcars/data/image/"
batches = jeremy_vgg16.Vgg16().get_batches(DATA_PATH+'train',
                                 gen=image.ImageDataGenerator(preprocessing_function=jeremy_vgg16.vgg_preprocess),
                                 batch_size=BATCH_SIZE)
val_batches = jeremy_vgg16.Vgg16().get_batches(DATA_PATH+'valid',
                                     gen=image.ImageDataGenerator(preprocessing_function=jeremy_vgg16.vgg_preprocess),
                                     batch_size=BATCH_SIZE)

input_layer = Input(shape=(3, 224, 224),
              name='image_input')
base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
x = base_model(input_layer)
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='softmax', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
predictions = Dense(batches.nb_class, activation='softmax', name='predictions')(x)

# this is the model we will train
keras_vgg = Model(input=input_layer, output=predictions)

# freeze all convolutional Vgg16 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
keras_vgg.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

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

