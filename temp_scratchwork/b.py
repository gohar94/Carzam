import os
import glob

import keras
from keras.preprocessing import image

from matplotlib import pyplot as plt
import numpy as np
from numpy.random import permutation
np.set_printoptions(precision=4, linewidth=100)
from sklearn.cross_validation import train_test_split
import skimage
from skimage import data, color, exposure

# Contains some utilty functions
# TODO Figure out what "reload" does
import utils; reload(utils)
from utils import plots
# Contains the trained Vgg16 model (2014 winner of ImageNet)
import vgg16; reload(vgg16)
from vgg16 import Vgg16

path = "../data/compcars/data/image/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

batch_size = 64

vgg = Vgg16()

batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size)

imgs, labels = next(batches)

vgg.finetune(batches)

vgg.model.load_weights(model_path+'compcars_finetune_100_epochs.h5')

history = vgg.fit(batches, val_batches, nb_epoch=50)

vgg.model.save_weights(model_path+'compcars_finetune_150_epochs.h5')

history = vgg.fit(batches, val_batches, nb_epoch=50)

vgg.model.save_weights(model_path+'compcars_finetune_200_epochs.h5')

history = vgg.fit(batches, val_batches, nb_epoch=50)

vgg.model.save_weights(model_path+'compcars_finetune_250_epochs.h5')

