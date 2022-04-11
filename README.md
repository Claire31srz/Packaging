# TP Packaging and Scripting
# TSE, M2 D3S
# Philippe and Serraz

This last TP is about packaging and scripting. We decided to go one with the TP on Lenet Like classifier and autoencoder as it is one of the most recent we've done and one that can be translated into Python. It is about image classification. This TP contains errors and is not over. We failed doing it completely as we didn't understand everything.


It is described as " This practical work is about creation of elementary models in keras.
While it focuses on image classification, it illustrates many important functionalities of the keras framework and tries to provide the user with good deep-learning development strategies.".

This package presents the classifier and the autoencoder.

 # About the Classifier
 
 The architecture has to be a CLASS called "Lenet_like".

- I can define the width: number of filters in the first layer.
- I can define the depth: number of conv layers.
- I can specify the number of classes: output neurons.

Lenet_like has no input tensor. But it has a \_\_call\_\_ function and can therefore be called on an input later.

The rule to go from depth d to depth d+1, is to reduce the spatial size by a factor of 2 in each direction.

Hidden dense layer will have 512 units.

The model needs to be able to fit on cifar10 dataset.

A function called "make_lenet_model"  is created to take the name of one of these dataset as a str.
It returns a keras Model object.
It should obviously take all arguments to init the Lenet_like architecture.
Arguments other than the dataset might have default values.
The accuraccy of the model is monitorable.

There is a function "fit_model_on" with the following arguments:

- dataset: str name of the dataset
- epochs: number of times you fit on the entire training set
- batch_size: number of images to average gradient on

The function must create a Lenet model and fit it following these parameters.
The function should:

- fit the model, obviously
- store the model architecture in a .json file in your output directory
- store the model's weights in a .h5 file in your output directory
- store the fitting metrics loss, validation_loss, accuracy, validation_accuracy under the form of a plot exported in a png file.

# About the autoencoder

Autoencoders are a very special kind of model. Input $x$ is an image, output $y$ is an image. The model is supposed to predict $x = y$ ! Usually, you build an autoencoder to create a 'deep' representation of your images : the model break your image into a high level descriptor that is useful to build your image back (the descriptors' space is supposed to be very powerful for clustering and image matching algorithms). The representation is built in an unsupervised way ; if $f$ is the function defined by your model, then you expect $f$ to be such as:
$$f(x) = y;$$
$$x = y$$
So you try to minimize $||f(x) - x||^2$ (mean squared error)

The architecture has to be a CLASS called "Autoencoder".

Basically, an autoencoder has two parts:
- the Encoder $(conv2D-Maxpooling)\times n$
- the Decoder $(conv2D-Upsampling)\times n$


Autoencoder has no input tensor. But it has a \_\_call\_\_ function and can therefore be called on an input later.

------------------------------

For the Encoder:

The rule to go from depth d to depth d+1, is to reduce the spatial size by a factor of 2 in each direction.

------------------------------
For the Decoder:

The rule to go from depth d to depth d+1, is to increase the spatial size by a factor of 2 in each direction.

The model is able to fit on the cifar-10 dataset.

 A function called "make_autoencoder_model" that take no argument is created.
It only returns a keras Model object.

Also, there are sseful functions to plot the resulting image produced by your autoencoder.
Read them carefully if you want to plot your output correctly (especially 'comparison')

The function "fit_model_cifar10" is createdwith the following arguments:

- dataset: str name of the dataset
- n_epochs: number of times you fit on the entire training set
- batch_size: number of images to average gradient on
- visualization_size (square root of number of test pred you wanna show to check your result)
- verbose (debug purpose to see tensor dimensions in your architecture)

The function must create an autoencoder model and fit it following these parameters.
The function does:

- fit the model, obviously
- store the model architecture in a .json file in your output directory
- store the model's weights in a .h5 file in your output directory
- store the fitting metrics loss, validation_loss, under the form of a plot exported in a png file.
- store the rebuilt test images under the form of a plot exported in a png file.

# Requirements

On the requirement file you file the three libraries mandatory with their version:

keras == 2.7.0
matplotlib == 3.0.2
numpy == 1.3.4

# Installation and import

You will need to pip install pathtopackage/LL but it won't work as there are still errors.
Then you can either import LL, either from LL import *

# Licence

The licence is given by Google.



