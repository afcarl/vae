'''Modified version of Keras' VAE example with deconvolution layers.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
import os

# input image dimensions
img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 100
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
epochs = 1

class VAE(object):

    def __init__(self, original_img_size, latent_dim, intermediate_dim, epsilon_std, epochs, batch_size):
        self.original_img_size = original_img_size
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epsilon_std = epsilon_std
        self.epochs = epochs
        self.batch_size = batch_size
        
        x, z_mean, z_log_var = self.build_encoder()
        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
            return z_mean + K.exp(z_log_var) * epsilon

        def vae_loss(x, x_decoded_mean_squash):
            x = K.flatten(x)
            x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
            xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)
        
        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_var])`
        self.z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        self.generator, x_decoded_mean_squash = self.build_decoder()

        # build a model to project inputs on the latent space
        self.encoder = Model(x, z_mean)     # where is z_var here?
        
        self.vae = Model(x, x_decoded_mean_squash)
        self.vae.compile(optimizer='rmsprop', loss=vae_loss) 
        
    def build_encoder(self):
        x = Input(batch_shape=(batch_size,) + original_img_size)
        conv_1 = Conv2D(img_chns,
                        kernel_size=(2, 2),
                        padding='same', activation='relu')(x)
        conv_2 = Conv2D(filters,
                        kernel_size=(2, 2),
                        padding='same', activation='relu',
                        strides=(2, 2))(conv_1)
        conv_3 = Conv2D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_2)
        conv_4 = Conv2D(filters,
                        kernel_size=num_conv,
                        padding='same', activation='relu',
                        strides=1)(conv_3)
        flat = Flatten()(conv_4)

        hidden = Dense(self.intermediate_dim, activation='relu')(flat)

        z_mean = Dense(self.latent_dim)(hidden)
        z_log_var = Dense(self.latent_dim)(hidden)

        return x, z_mean, z_log_var

    def build_decoder(self):
        # we instantiate these layers separately so as to reuse them later
        
        decoder_hid = Dense(intermediate_dim, activation='relu')
        decoder_upsample = Dense(filters * 14 * 14, activation='relu')

        if K.image_data_format() == 'channels_first':
            output_shape = (batch_size, filters, 14, 14)
        else:
            output_shape = (batch_size, 14, 14, filters)

        decoder_reshape = Reshape(output_shape[1:])
        decoder_deconv_1 = Conv2DTranspose(filters,
                                           kernel_size=num_conv,
                                           padding='same',
                                           strides=1,
                                           activation='relu')
        decoder_deconv_2 = Conv2DTranspose(filters, num_conv,
                                           padding='same',
                                           strides=1,
                                           activation='relu')
        if K.image_data_format() == 'channels_first':
            output_shape = (batch_size, filters, 29, 29)
        else:
            output_shape = (batch_size, 29, 29, filters)
        decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                                  kernel_size=(3, 3),
                                                  strides=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        decoder_mean_squash = Conv2D(img_chns,
                                     kernel_size=2,
                                     padding='valid',
                                     activation='sigmoid')

        
        hid_decoded = decoder_hid(self.z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
        
        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim,))
        _hid_decoded = decoder_hid(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
        generator = Model(decoder_input, _x_decoded_mean_squash)

        return generator, x_decoded_mean_squash 


    def train(self, x_train, x_test, y_test):
        self.vae.fit(x_train, x_train, shuffle=True, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test, x_test))

    def save(self, output_dir):
        self.vae.save_weights(os.path.join(output_dir, 'vae_model.h5'))
        self.encoder.save_weights(os.path.join(output_dir, 'encoder_model.h5'))
        self.generator.save_weights(os.path.join(output_dir, 'generator_model.h5'))
        
    def load(self, output_dir):
        vae_filename = os.path.join(output_dir, 'vae_model.h5')
        encoder_filename = os.path.join(output_dir, 'encoder_model.h5')
        generator_filename = os.path.join(output_dir, 'generator_model.h5')
        
        self.encoder.load_weights(encoder_filename)
        self.vae.load_weights(vae_filename)
        self.generator.load_weights(generator_filename)
        

if __name__ == "__main__":
    
    vae = VAE(original_img_size, latent_dim, intermediate_dim, epsilon_std, epochs, batch_size)
    
    # train the VAE on MNIST digits
    (x_train, _), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)
    print('x_train.shape:', x_train.shape)

    vae.train(x_train, x_test, y_test)
    
    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = vae.encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

    
    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = vae.generator.predict(z_sample, batch_size=batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
