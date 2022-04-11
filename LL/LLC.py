# Part 0: Libraries

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Input
from tensorflow.keras.utils import to_categorical
import numpy
import pandas as pd
import os
from tensorflow.keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt

# Part 1: Class Lenet Like Definition

class Lenet_like:
  """
  Lenet like architecture.
  """
  def __init__(self, width, depth, drop, n_classes):
    """
    Architecture settings.

    Arguments:
      - width: int, first layer number of convolution filters.
      - depth: int, number of convolution layer in the network.
      - drop: float, dropout rate.
      - n_classes: int, number of classes in the dataset.
    """
    self.width = width
    self.depth = depth
    self.drop = drop
    self.n_classes = n_classes
  
  def __call__(self, X):
    """
    Call classifier layers on the inputs.
    """

    for k in range(self.depth):
      # Apply successive convolutions to the input !
      # Use the functional API to do so

      if k == 0 : 
        Y = Conv2D(self.width*(2**k), self.depth, padding="same", activation="relu")(X)
        Y = Conv2D(self.width*(2**k), self.depth, padding="same", activation="relu")(Y)
        Y = MaxPooling2D()(Y)

      else: 
        Y = Conv2D(self.width*(2**k), self.depth, padding="same", activation="relu")(Y)
        Y = Conv2D(self.width*(2**k), self.depth, padding="same", activation="relu")(Y)
        Y = MaxPooling2D()(Y)

    # Perceptron
    # This is the classification head of the classifier
    Y = Dropout(self.drop)(Y)
    Y = Dense(512)(Y)
    Y = Flatten()(Y)
    Y = Dropout(self.drop)(Y)
    Y = Dense(self.n_classes, activation='softmax')(Y)

    return Y

# Part 2: Function creation to design model
def make_lenet_model(dataset,
                     width=32,
                     depth=2,
                     drop=0.25,
                     n_classes=10):
  """
  Create a Lenet model adapted to the dimensions of a given dataset.
  """
  if dataset == "cifar10":
    # dimensions of input are: (32, 32, 3)
    X = Input(batch_shape=(None, 32, 32, 3))
  elif dataset == "mnist" or dataset == "fashion_mnist":
    # dimensions of input are: (28, 28)
    X = Input(batch_shape=(None, 28,28, 1))
  else:
    raise NotImplementedError("Model not implemented for datastet {}".format(dataset))
  
  Y = Lenet_like(width, depth, drop, n_classes)(X)
  
  model = Model(inputs=X, outputs=Y)
  # Remember I wanna monitor accuracy
  model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
  return model

# Part 3: Function creation to fit model

def fit_model_on(dataset,
                 epochs=100,
                 batch_size=32,
                 n_classes=10):
  
  # create your model and call it on your dataset
  model = make_lenet_model(dataset, width = batch_size, depth = 2, drop =0.2, n_classes=n_classes)
  # create a Keras ImageDataGenerator to handle your dataset
  datagen = ImageDataGenerator(
        width_shift_range=3 /32, #dimensions of input are: (32, 32, 3) so we need to divide 32 per 3 
        height_shift_range=3 /32, #dimensions of input are: (32, 32, 3) so we need to divide 32 per 3 
        horizontal_flip=True)
  
  if dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  elif dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
  elif dataset == "fashion_mnist":
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  else:
    raise NotImplementedError("Model not implemented for datastet {}".format(dataset))
  
  # Convert class vectors to binary class matrices (one-hot encoding).
  y_train = to_categorical(y_train, n_classes)
  y_test = to_categorical(y_test, n_classes)

  # Be sure that your training/test data is 'float32'
  x_train = x_train.astype(float)
  x_test = x_test.astype(float)
  # Be sure that your training/test data are between 0 and 1 (pixel image value)
  x_train = x_train/255
  x_test = x_test/255

  try:
    # Fit with keras using 'datagen', the previously defined image generator
    datagen.fit(x_train)
    # We don't want out data to shuffle
    history = model.fit(datagen.flow(x_train, y_train, shuffle = False),  
                        batch_size = batch_size,  
                        steps_per_epoch = len(x_train)//batch_size, 
                        epochs = epochs,
                        validation_data = datagen.flow(x_test, y_test, shuffle = False) )
  except KeyboardInterrupt:
    print("Training interrupted!")
  
  
  # finally, plot and save the metrics
  plt.plot(history.history['accuracy'], label= 'train set')
  plt.plot(history.history['val_accuracy'], label='validation set')
  plt.legend()
  plt.title('Evolution of Accuracy for both sets')
  plt.show()

  plt.plot(history.history['loss'], label= 'train set')
  plt.plot(history.history['val_loss'], label='validation set')
  plt.legend()
  plt.title('Evolution of Loss for both sets')
  plt.show()
  
  return modem