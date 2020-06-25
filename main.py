importnumpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPoolig2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimisers import Adam
from tensorflow.keras.callbacaks import ModdelCheckpoint, ReduceROnPlateau
from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image
from livelossplot import PlotLossesTensorFlowKeras
import tensorflow as tf
print(" Tensorflow version:", tf.__version__ )


img_size=48
batch_size= 64
datagen_train = ImageDataGenerator( horizontal_flip=True )		#img augmentaion
train_generator = datagen_train.flow_from_directory("train/", target_size =(img_size, img_size), color_mode= "grayscale", batch_size=batch_size, class_mode= "categorical", shuffle= True) 

datagen_validation= ImageDataGenerator( horizontal_flip=True )	#img augmentaion
train_generator = datagen_train.flow_from_directory("test/", target_size =(img_size, img_size), color_mode= "grayscale", batch_size=batch_size, class_mode= "categorical", shuffle= True) 
