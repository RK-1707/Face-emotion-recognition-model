import numpy as np
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
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
print(" Tensorflow version:", tf.__version__ )


img_size=48
batch_size= 64
datagen_train = ImageDataGenerator( horizontal_flip=True )		#img augmentaion
train_generator = datagen_train.flow_from_directory("train/", target_size =(img_size, img_size), color_mode= "grayscale", batch_size=batch_size, class_mode= "categorical", shuffle= True) 

datagen_validation= ImageDataGenerator( horizontal_flip=True )	#img augmentaion
train_generator = datagen_train.flow_from_directory("test/", target_size =(img_size, img_size), color_mode= "grayscale", batch_size=batch_size, class_mode= "categorical", shuffle= True) 

#model generation and fitting 
model = Sequential()

#1st conv2d layer
model.add(Conv2D(64, (3,3), padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization() )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#2nd conv2d layer
model.add(Conv2D(128, (5,5), padding='same'))
model.add(BatchNormalization() )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#3rd conv2d layer
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization() )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#4th conv2d layer
model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization() )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten() )

model.add(Dense(256) )
model.add(BatchNormalization() )
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512) )
model.add(BatchNormalization() )
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation= 'softmax'))

opt=Adam( lr=0.0005 )
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']
model.summary()

#training and evaluate model
epochs = 50		#increase to get better acc
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

checkpoint = modelCheckpoint("model_weights.h5",
                             moniter='val_accuracy',
                             save_weights_only=True,
                             mode='max',
                             verbose=1)
reduce_lr = ReduceLROnPlateau(moniter='val_loss',
                              factor=0.1,
                              patience=2,
                              min_lr= 0.00001,
                              model='auto')

callbacks = [PlotLossesTensorflowKeras(), checkpoint, reduce_lr]
history = model.fit(
	x= train_generator,
	steps_per_epoch= steps_per_epoch,
	epochs= epochs,
	validation_data= validation_generator,
	validation_steps= validation_steps,
	callbacks= callbacks
)


#represent model as JSON string
model_json=model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)


	      
# plot the evolution of Loss and Acuracy on the train and validation sets
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
	      
	      
# show the confusion matrix of our predictions
# compute predictions
predictions = model.predict_generator(generator=validation_generator)
y_pred = [np.argmax(probas) for probas in predictions]
y_test = validation_generator.classes
class_names = validation_generator.class_indices.keys()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Normalized confusion matrix')
plt.show()
