from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import datetime
import matplotlib.pyplot as plt
from utils import DataLoader, time, make_trainable
import numpy as np
import os

import keras.backend as K

class DCGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.lr_height = 64                 # Low resolution height
        self.lr_width = 64                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*2   # High resolution height
        self.hr_width = self.lr_width*2     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.latent_dim = 100
        self.time = time()

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        optimizer = Adam(0.00005, 0.9)
        """
        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        """
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        print("---------------------vgg summary----------------------------")
        self.vgg.summary()
        self.vgg.compile(loss='mse',
                         optimizer=optimizer)

        # Configure data loader
        self.dataset_name = 'celebA'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))
        self.n_data = self.data_loader.get_n_data()

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        self.generator = self.build_generator()
        print("---------------------generator summary----------------------------")
        self.generator.summary()

        self.generator.compile(loss='ssim',
                               optimizer=optimizer,
                               metrics=['mse'])

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        print("\n---------------------discriminator summary----------------------------")
        self.discriminator.summary()

        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        make_trainable(self.discriminator, False)

        # High res. and low res. images
        img_hr = Input(shape=(224, 224, self.channels))
        img_lr = Input(shape=self.lr_shape)

        z = Input(shape=(self.latent_dim,))
        fake_lr = self.generator(z)

        # for the combined model, we only train ganerator
        self.discriminator.trainable = False

        validity = self.discriminator(fake_lr)

        fake_features = self.vgg(fake_lr)

        self.combined = Model([z], [validity, fake_features])
        print("\n---------------------combined summary----------------------------")
        self.combined.summary()
        self.combined.compile(loss=['binary_crossentropy', 'ssim'],
                              loss_weights=[1e-2, 1],
                              optimizer=optimizer)

        self.dLosses = []
        self.gLosses = []


    def build_generator(self):
        noise = Input(shape=(self.latent_dim,))

        def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(2, 2), bn=True):
            """Layers used during upsampling"""
            #u = UpSampling2D(size=2)(layer_input)
            u = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
            if bn:
                u = BatchNormalization(momentum=0.8)(u)
            u = Activation('relu')(u)
            return u

        generator = Dense(4*4*self.gf*8, activation="relu")(noise)
        generator = Reshape((4, 4, self.gf*8))(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = deconv2d(generator, filters=self.gf*8, kernel_size=(5, 5), strides=(2, 2))
        generator = deconv2d(generator, filters=self.gf*4, kernel_size=(5, 5), strides=(2, 2))
        generator = deconv2d(generator, filters=self.gf*2, kernel_size=(5, 5), strides=(2, 2))
        generator = deconv2d(generator, filters=self.gf, kernel_size=(5, 5), strides=(2, 2), bn=False)

        gen_img = Conv2D(self.channels, kernel_size=3, padding="same", activation='tanh')(generator)

        return Model(noise, gen_img)

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        inp = Input(shape=(None, None, self.channels))
        try:
            img = Lambda(lambda image: K.tf.image.resize_images(image, (224, 224)))(inp)
        except:
            img = Lambda(lambda image: K.tf.image.resize_images(image, 224, 224))(inp)

        # Extract image features
        img_features = vgg(img)

        return Model(inp, img_features)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)

            return d

        # Input img = generated image
        d0 = Input(shape=self.lr_shape)

        d = d_block(d0, self.df, strides=2, bn=False)
        d = d_block(d, self.df*2, strides=2)
        d = d_block(d, self.df*4, strides=2)
        d = d_block(d, self.df*8, strides=2)

        d = Dense(self.df*16)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Flatten()(d)
        validity = Dense(1, activation='sigmoid')(d)

        return Model(d0, validity)

    def train(self, epochs, batch_size, sample_interval):

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        max_iter = int(self.n_data/batch_size)
        print("\nbatch size : %d | num_data : %d | max iteration : %d \n" % (batch_size, self.n_data, max_iter))
        for epoch in range(1, epochs+1):
            for iter in range(max_iter):
                # ------------------
                #  Train Generator
                # ------------------
                imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                make_trainable(self.discriminator, True)
                d_loss_real = self.discriminator.train_on_batch(imgs_lr, valid*0.9)  # label smoothing
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                make_trainable(self.discriminator, False)
                image_features = self.vgg.predict(imgs_lr)

                if iter % (sample_interval/10) == 0:
                    tensorboard = TensorBoard('./logs/%s' % self.time)
                    tensorboard.set_model(self.generator)

                g_loss = self.combined.train_on_batch([noise], [valid, image_features])

                if iter % 10 == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    print("epoch:%d | iter : %d / %d | time : %10s | g_loss : %15s | d_loss : %s " %
                          (epoch, iter, max_iter, elapsed_time, g_loss, d_loss))

                if (iter+1) % sample_interval == 0:
                    self.sample_images(epoch, iter+1)

            self.dLosses.append(d_loss[0] if type(d_loss) is list else d_loss)
            self.gLosses.append(g_loss[0] if type(g_loss) is list else g_loss)

            # save model
            self.save_model(self.generator, epoch, name='generator')

    def save_model(self, model, epoch, name='model'):
        os.makedirs('models/%s' % self.time, exist_ok=True)
        model.save_weights('models/%s/%s_epoch%d_weights.h5' % (self.time, name, epoch))
        with open('models/%s/%s_epoch%d_architecture.json' % (self.time, name, epoch), 'w') as f:
            f.write(model.to_json())

    def sample_images(self, epoch, iter):
        os.makedirs('samples/%s' % self.time, exist_ok=True)

        r, c = 5, 5

        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        gen_img = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_img = 0.5 * gen_img + 0.5

        # Save generated images and the high resolution originals
        fig, axs = plt.subplots(r, c)
        for row in range(r):
            for col in range(c):
                axs[row, col].imshow(gen_img[5*row+col])
                axs[row, col].axis('off')
        fig.savefig("samples/%s/e%d-i%d.png" % (self.time, epoch, iter), bbox_inches='tight', dpi=100)
        plt.close()

if __name__ == '__main__':
    gan = DCGAN()
    gan.train(epochs=10, batch_size=32, sample_interval=400)
