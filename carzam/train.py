# TODO Add shebang
"""

"""
from keras.preprocessing import image
import compcars_models
import utils

def train(model_name):
    """
    # TODO Docstring.
    """
    img_rows = 224
    img_cols = 224
    channel = 3
    batch_size = 64
    data_path = "../data/compcars/data/image/"
    imagenet_model_path = "../imagenet_models/"
    
    # Get images
    batches = utils.get_batches(data_path+'train', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=batch_size)
    val_batches = utils.get_batches(data_path+'valid', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=batch_size)
    
    # Create model
    if model_name == "inception_v1":
        model = compcars_models.googlenet_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    elif model_name == "vgg19":
        model = compcars_models.vgg19_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    
    # Train and save intermediate results
    history = model.fit_generator(batches,
                            validation_data=val_batches,
                            samples_per_epoch=batches.nb_sample,
                            nb_val_samples=val_batches.nb_sample,
                            nb_epoch=50)
    model.save_weights(model_name+'_50.h5')

    history = model.fit_generator(batches,
                            validation_data=val_batches,
                            samples_per_epoch=batches.nb_sample,
                            nb_val_samples=val_batches.nb_sample,
                            nb_epoch=50)
    model.save_weights(model_name+'_100.h5')

if __name__ == '__main__':
    train("inception_v1")
