# Part 0: Libraries to import

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import BatchNormalization, UpSampling2D
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, Input
from tensorflow.keras.utils import to_categorical
import numpy
import pandas as pd
import os
from tensorflow.keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt


# Part 1: Class creation
# first model
# Lenet-like
class Autoencoder:
  """
  Aurtoencoder architecture.
  """
  def __init__(self):
    """
    Architecture settings.
    """
    # nothing to do in the init.
  
  def __call__(self, X):
    """
    Call autoencoder layers on the inputs.
    """

    # encode
    Y = Conv2D(32, kernel_size=(3,3), strides=2, padding='same', activation='relu')(MaxPooling2D()(X))
    Y = Conv2D(32, kernel_size=(3,3), strides=1, padding='same', activation='relu')(Y)
    Y = BatchNormalization()(Y)
    # decode
    Y = Conv2D(32, kernel_size=(3,3), strides=1, padding='same', activation='relu')(UpSampling2D()(Y))
    Y = BatchNormalization()(Y)
    Y = Conv2D(3, kernel_size=1, padding='same', activation='sigmoid')(UpSampling2D()(Y))

    return Y

# Part 2: Function creation to make model
    
def make_autoencoder_model():
  """
  Create and compile autoencoder keras model.
  """
  X = Input(batch_shape=(None, 32, 32, 3))
  Y = Autoencoder()(X)
  model = Model(inputs=X, outputs=Y)
  model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
  return model

# Part 3: Functions to plot the resulting image
  
def stack_horizontally(min, max, images):
    return numpy.hstack(images[i] for i in range(min, max))

def stack_vertically(length, height, images):
    return numpy.vstack(stack_horizontally(i * length, (i + 1) * length, images) for i in range(height))

def comparison(inputimgs, outputimgs, length, height):
    A = stack_vertically(length, height, inputimgs)
    B = stack_vertically(length, height, outputimgs)
    C = numpy.ones((A.shape[0], 32, 3))
    return numpy.hstack((A, C, B))

# 4. Part 4: Fitting function creation
def fit_model_on_cifar10(n_epochs=3, batch_size=128, visualization_size=5, verbose=1):

  # create your model and call it on your dataset
  model = make_autoencoder_model()
  if verbose > 0:
    print(model.summary())
  (x_train, _), (x_test, _) = cifar10.load_data()
  # Be sure that your training/test data is 'float32'
  x_train = x_train.astype(float)
  x_test = x_test.astype(float)
  # Be sure that your training/test data are between 0 and 1 (pixel image value)
  x_train = x_train / 255
  x_test = x_test / 255
  try:
    history = model.fit(x_train, x_train,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_test, x_test))
  except KeyboardInterrupt:
    print("Training interrupted!")
  
  # now, save metrics plots
  plt.plot(history.history['loss'], label= 'train set')
  plt.plot(history.history['val_loss'], label='validation set')
  plt.legend()
  plt.title('Evolution of Loss for both sets')
  plt.show()

  prediction= model.predict(x_test)
  ipt=[]
  otp=[]

  plt.imshow(comparison(x_test,prediction, visualization_size, visualization_size ))
  plt.show()
  return model
  
  