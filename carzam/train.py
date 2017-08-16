# TODO Add shebang
"""

"""
from keras.preprocessing import image
from compcars_vgg19_model import vgg19_model
from compcars_inception_v1_model import googlenet_model
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
    batches = utils.get_batches(data_path+'train', 
            gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess, 
                rotation_range=10, 
                width_shift_range=0.1, 
                height_shift_range=0.1, 
                shear_range=0.15,
                zoom_range=0.1,
                channel_shift_range=10.,
                horizontal_flip=True), 
            batch_size=batch_size)
    val_batches = utils.get_batches(data_path+'valid', gen=image.ImageDataGenerator(preprocessing_function=utils.vgg_preprocess), batch_size=batch_size)
    
    # Create model
    if model_name == "inception_v1":
        model = googlenet_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    elif model_name == "vgg19":
        model = vgg19_model(img_rows, img_cols, channel, batches.nb_class, imagenet_model_path)
    
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
