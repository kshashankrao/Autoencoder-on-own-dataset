# Autoencoder-on-own-dataset
This repository contains script to train and test an autoencoder using Keras.
It supports training on own dataset.

Autoencoders is a type of unsupervised learning used to represent a set of data. The concept is to learn the features of input and map the output as close to the input by reconstruction. To accomplish it, encoder and decoder are used.

Autoencoders is a type of unsupervised learning used to represent a set of data. The concept is to learn the features of input and map the output as close to the input by reconstruction. To accomplish it, encoder and decoder are used.

The architecture of an autoencoder mainly consists of encoder and decoder. Encoder is a neural network consisting of hidden layers that extracts the features of the image. Usually the dimension of the feature map after each hidden layer reduces. In other words, compression of input image occurs at this stage. At the latent layer(last layer of encoder), the output is flattened and reshaped. This is a hyper parameter.

The decoder reconstructs the image. Its architecture is mirror image of the encoder i.e it upsamples the data fed to it. The process of reconstruction involves copying the input image. This is useful in application such as denoising where the model would have been trained on clean image and is used to remap the corrupted images.

One of the important rule to follow while designing auto encoders is that the input and output layers should have same shape unlike supervised learning.

Reference:
https://greengnomie.wordpress.com/2018/08/20/autoencoder/
