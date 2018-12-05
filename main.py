from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import datetime
import matplotlib.pyplot as plt
from utils import DataLoader, time, make_trainable
import numpy as np
import os

DEBUG = 0

class DCGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.img_size = 64
        self.latent_dim = 100
        self.time = time()
        self.dataset_name = 'vdsr'
        self.learning_rate = 1e-4

        optimizer = Adam(self.learning_rate, beta_1=0.5, decay=0.00005)

        self.gf = 64 # filter size of generator's last layer
        self.df = 64 # filter size of discriminator's first layer

        # Configure data loader
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_size, self.img_size), mem_load=True)
        self.n_data = self.data_loader.get_n_data()



        self.generator = self.build_generator()
        print("---------------------generator summary----------------------------")
        self.generator.summary()

        self.generator.compile(loss='mse',
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

        z = Input(shape=(self.latent_dim,))
        fake_img = self.generator(z)

        # for the combined model, we only train ganerator
        self.discriminator.trainable = False

        validity = self.discriminator(fake_img)

        self.combined = Model([z], [validity])
        print("\n---------------------combined summary----------------------------")
        self.combined.summary()
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def build_generator(self):
        noise = Input(shape=(self.latent_dim,))

        def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(2, 2), bn_relu=True):
            """Layers used during upsampling"""
            # u = UpSampling2D(size=2)(layer_input)
            u = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
            if bn_relu:
                u = BatchNormalization(momentum=0.9)(u)
                u = Activation('relu')(u)
            return u

        generator = Dense(16 * self.gf * self.img_size // 16 * self.img_size // 16, activation="relu")(noise)
        generator = Reshape((self.img_size // 16, self.img_size // 16, self.gf * 16))(generator)
        generator = BatchNormalization()(generator)
        generator = Activation('relu')(generator)
        generator = deconv2d(generator, filters=self.gf * 8)
        generator = deconv2d(generator, filters=self.gf * 4)
        generator = deconv2d(generator, filters=self.gf * 2)
        generator = deconv2d(generator, filters=self.gf    )
        generator = deconv2d(generator, filters=self.channels, kernel_size=(3,3), strides=(1,1), bn_relu=False)

        gen_img = Activation('tanh')(generator)

        return Model(noise, gen_img)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.9)(d)
            d = LeakyReLU(alpha=0.2)(d)

            return d

        # Input img = generated image
        d0 = Input(shape=(self.img_size, self.img_size, self.channels))

        d = d_block(d0, self.df, strides=2, bn=False)
        d = d_block(d, self.df*2, strides=2)
        d = d_block(d, self.df*4, strides=2)
        d = d_block(d, self.df*8, strides=2)

        d = Flatten()(d)
        validity = Dense(1, activation='sigmoid')(d)

        return Model(d0, validity)

    def train(self, epochs, batch_size, sample_interval):
        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        max_iter = int(self.n_data/batch_size)
        os.makedirs('./logs/%s' % self.time, exist_ok=True)
        tensorboard = TensorBoard('./logs/%s' % self.time)
        tensorboard.set_model(self.generator)

        os.makedirs('models/%s' % self.time, exist_ok=True)
        with open('models/%s/%s_architecture.json' % (self.time, 'generator'), 'w') as f:
            f.write(self.generator.to_json())
        print("\nbatch size : %d | num_data : %d | max iteration : %d | time : %s \n" % (batch_size, self.n_data, max_iter, self.time))
        for epoch in range(1, epochs+1):
            for iter in range(max_iter):
                # ------------------
                #  Train Generator
                # ------------------
                ref_imgs = self.data_loader.load_data(batch_size)

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                make_trainable(self.discriminator, True)
                d_loss_real = self.discriminator.train_on_batch(ref_imgs, valid*0.9)  # label smoothing
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                make_trainable(self.discriminator, False)

                logs = self.combined.train_on_batch([noise], [valid])
                tensorboard.on_epoch_end(iter, named_logs(self.combined, [logs]))

                if iter % (sample_interval // 10) == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    print("epoch:%d | iter : %d / %d | time : %10s | g_loss : %15s | d_loss : %s " %
                          (epoch, iter, max_iter, elapsed_time, logs, d_loss))

                if (iter+1) % sample_interval == 0:
                    self.sample_images(epoch, iter+1)

            # save weights after every epoch
            self.generator.save_weights('models/%s/%s_epoch%d_weights.h5' % (self.time, 'generator', epoch))

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
    if DEBUG == 1:
        gan.n_data = 50
        gan.train(epochs=2, batch_size=1, sample_interval=10)
    else:
        gan.train(epochs=10, batch_size=64, sample_interval=200)
