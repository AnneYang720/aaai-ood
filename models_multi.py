from __future__ import print_function, division
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import keras.backend as K
import cv2
import logging
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import *
from kh_tools import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


class ALOCC_Model():
    def __init__(self, input_height=28,input_width=28, output_height=28, output_width=28,
               attention_label=[1], class_num=None, is_training=True, z_dim=100, gf_dim=16, df_dim=16, c_dim=3,
               dataset_name=None, dataset_address=None, input_fname_pattern=None,
               checkpoint_dir='checkpoint_multi', log_dir='log_multi', sample_dir='sample_multi', r_alpha = 0.2,
               kb_work_on_patch=True, nd_patch_size=(10, 10), n_stride=1,
               n_fetch_data=10):
        """
        This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection.
        :param sess: TensorFlow session.
        :param input_height: The height of image to use.
        :param input_width: The width of image to use.
        :param output_height: The height of the output images to produce.
        :param output_width: The width of the output images to produce.
        :param attention_label: Conditioned label that growth attention of training label [1]
        :param is_training: True if in training mode.
        :param z_dim:  (optional) Dimension of dim for Z, the output of encoder. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer, i.e. g_decoder_h0. [16]
        :param df_dim: (optional) Dimension of discrim filters in first conv layer, i.e. d_h0_conv. [16]
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param dataset_name: 'UCSD', 'mnist' or custom defined name.
        :param dataset_address: path to dataset folder. e.g. './dataset/mnist'.
        :param input_fname_pattern: Glob pattern of filename of input images e.g. '*'.
        :param checkpoint_dir: path to saved checkpoint(s) directory.
        :param log_dir: log directory for training, can be later viewed in TensorBoard.
        :param sample_dir: Directory address which save some samples [.]
        :param r_alpha: Refinement parameter, trade-off hyperparameter for the G network loss to reconstruct input images. [0.2]
        :param kb_work_on_patch: Boolean value for working on PatchBased System or not, only applies to UCSD dataset [True]
        :param nd_patch_size:  Input patch size, only applies to UCSD dataset.
        :param n_stride: PatchBased data preprocessing stride, only applies to UCSD dataset.
        :param n_fetch_data: Fetch size of Data, only applies to UCSD dataset.
        """

        self.b_work_on_patch = kb_work_on_patch
        self.sample_dir = sample_dir
        self.is_training = is_training
        self.r_alpha = r_alpha
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim
        self.dataset_name = dataset_name
        self.dataset_address= dataset_address
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.attention_label = attention_label

        if self.is_training:
            logging.basicConfig(filename='ALOCC_loss.log', level=logging.INFO)

        if self.dataset_name == 'mnist':
            self.c_dim = 1
            self.class_num = len(self.attention_label)
            (X_train, y_train), (_, _) = mnist.load_data()
            # Make the data range between 0~1.
            X_train = X_train / 255
            self.data = []
            self.label = []
            for i in range(len(self.attention_label)):
                specific_idx = np.where(y_train == self.attention_label[i])[0]
                for j in specific_idx:
                    self.data.append(X_train[j].reshape(28, 28, 1))
                    self.label.append(i)
            self.label = to_categorical(self.label, num_classes=self.class_num)
            para = np.arange(len(self.data))
            np.random.shuffle(para)
            self.data = np.array(self.data)[para]
            self.label = np.array(self.label)[para]
            print('data shape',self.data.shape)
            print('label shape',self.label.shape)
        elif self.dataset_name == 'cifar10':
            self.c_dim = 3
            self.class_num = len(self.attention_label)
            (X_train, y_train), (_, _) = cifar10.load_data()
            print('X_train shape',X_train.shape)
            print('y_train shape',y_train.shape)
            # Make the data range between 0~1.
            X_train = X_train / 255
            self.data = []
            self.label = []
            for i in range(len(self.attention_label)):
                specific_idx = np.where(y_train == self.attention_label[i])[0]
                for j in specific_idx:
                    self.data.append(X_train[j].reshape(32, 32, 3))
                    self.label.append(i)
            self.label = to_categorical(self.label, num_classes=self.class_num)
            #self.class_num = self.label.shape[1]
            #self.label = to_categorical(y_train, num_classes=self.class_num+1)
            para = np.arange(len(self.data))
            np.random.shuffle(para)
            self.data = X_train[para]
            self.label = self.label[para]
            print('data shape',self.data.shape)
            print('label shape',self.label.shape)
        else:
            assert('Error in loading dataset')

        self.input_height = self.data.shape[1]
        self.input_width = self.data.shape[2]
        self.output_height = self.data.shape[1]
        self.output_width = self.data.shape[2]
        self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_generator(self, input_shape):
        """Build the generator/R network.

        Arguments:
            input_shape {list} -- Generator input shape.

        Returns:
            [Tensor] -- Output tensor of the generator/R network.
        """
        image = Input(shape=input_shape, name='z')

        # Encoder.
        x = Conv2D(name='g_encoder_h0_conv', filters=self.df_dim * 2, kernel_size = 3, strides=2, padding='same')(image)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(name='g_encoder_h1_conv', filters=self.df_dim * 4, kernel_size = 3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(name='g_encoder_h2_conv', filters=self.df_dim * 8, kernel_size = 3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decoder-version1.
        # TODO: need a flexable solution to select output_padding and padding.
        x = Conv2DTranspose(self.gf_dim*2, kernel_size = 3, strides=2, activation='relu', padding='same', output_padding=0, name='g_decoder_h0')(x)
        x = BatchNormalization()(x)
        if self.dataset_name == 'cifar10':
            x = Conv2DTranspose(self.gf_dim*2, kernel_size=2, strides=1, name='g_decoder_adjust')(x)
            x = BatchNormalization()(x)
        x = Conv2DTranspose(self.gf_dim*1, kernel_size = 3, strides=2, activation='relu', padding='same', output_padding=1, name='g_decoder_h1')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(self.c_dim, kernel_size = 3, strides=2, activation='tanh', padding='same', output_padding=1, name='g_decoder_h2')(x)

        return Model(image, x, name='R')

    def build_discriminator(self, input_shape, class_num):
        """Build the discriminator/D network

        Arguments:
            input_shape {list} -- Input tensor shape of the discriminator network, either the real unmodified image
                or the generated image by generator/R network.

        Returns:
            [Tensor] -- Network output tensors.
        """

        image = Input(shape=input_shape, name='d_input')
        x = Conv2D(name='d_h0_conv', filters=self.df_dim*1, kernel_size = 5, strides=1, padding='same')(image)
        x = LeakyReLU()(x)

        x = Conv2D(name='d_h1_conv', filters=self.df_dim*2, kernel_size = 5, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(name='d_h2_conv', filters=self.df_dim*4, kernel_size = 5, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(name='d_h3_conv', filters=self.df_dim*8, kernel_size = 5, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(name='d_h3_output', units=class_num, activation='softmax')(x)

        return Model(image, x, name='D')

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        optimizer = RMSprop(lr=0.0002, clipvalue=1.0, decay=1e-8)
        # Construct discriminator/D network takes real image as input.
        # D - sigmoid and D_logits -linear output.
        self.discriminator = self.build_discriminator(image_dims, self.class_num)

        # Model to train D to discrimate real images.
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')#loss='categorical_crossentropy')

        # Construct generator/R network.
        self.generator = self.build_generator(image_dims)
        img = Input(shape=image_dims)

        reconstructed_img = self.generator(img)

        self.discriminator.trainable = False
        validity = self.discriminator(reconstructed_img)

        # Model to train Generator/R to minimize reconstruction loss and trick D to see
        # generated images as real ones.
        self.adversarial_model = Model(img, [reconstructed_img, validity])
        self.adversarial_model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[self.r_alpha, 1], optimizer=optimizer)

        print('\n generator')
        self.generator.summary()

        print('\n discriminator')
        self.discriminator.summary()

        print('\n adversarial_model')
        self.adversarial_model.summary()


    def train(self, epochs, batch_size = 128, sample_interval=500):
        # Make log folder if not exist.
        log_dir = os.path.join(self.log_dir, self.model_dir)
        os.makedirs(log_dir, exist_ok=True)

        sample = self.data[0:batch_size]

        # Export images as montage, sample_input also use later to generate sample R network outputs during training.
        sample_inputs = np.array(sample).astype(np.float32)
        print('sample_inputs shape', sample_inputs.shape)
        os.makedirs(self.sample_dir, exist_ok=True)
        cv2.imwrite('./{}/{}_train_input_samples.jpg'.format(self.sample_dir, self.dataset_name), montage(sample_inputs[:,:,:,0]))

        counter = 1
        # Record generator/R network reconstruction training losses.
        plot_epochs = []
        plot_g_recon_losses = []

        # Load traning data, add random noise.
        sample_w_noise = get_noisy_data(self.data)

        # Adversarial ground truths
        fake_labels = np.zeros((batch_size, self.class_num))
        #fake_labels = np.concatenate((fake_labels, np.ones((batch_size,1))), axis=1)

        for epoch in range(epochs):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch,epochs))
            # Number of batches computed by total number of target data / batch size.
            batch_idxs = len(self.data) // batch_size

            for idx in range(batch_idxs):
                # Get a batch of images and add random noise.
                batch = self.data[idx * batch_size:(idx + 1) * batch_size]
                label = self.label[idx * batch_size:(idx + 1) * batch_size]
                batch_noise = sample_w_noise[idx * batch_size:(idx + 1) * batch_size]
                batch_clean = self.data[idx * batch_size:(idx + 1) * batch_size]
                # Turn batch images data to float32 type.
                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = np.array(batch_noise).astype(np.float32)
                batch_clean_images = np.array(batch_clean).astype(np.float32)
                batch_fake_images = self.generator.predict(batch_images)
                # Update D network, minimize real images inputs->D->k-ones, noisy z->R->D->k+1-one loss.
                d_loss_real = self.discriminator.train_on_batch(batch_images, label)
                d_loss_fake = self.discriminator.train_on_batch(batch_fake_images, fake_labels)

                # Update R network twice, minimize noisy z->R->D->ones and reconstruction loss.
                self.adversarial_model.train_on_batch(batch_images, [batch_clean_images, label])
                g_loss = self.adversarial_model.train_on_batch(batch_images, [batch_clean_images, label])
                plot_epochs.append(epoch+idx/batch_idxs)
                plot_g_recon_losses.append(g_loss[1])
                counter += 1
                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}, g_loss:{4:>0.3f}, g_recon_loss:{4:>0.3f}'.format(epoch, idx, batch_idxs, d_loss_real+d_loss_fake, g_loss[0], g_loss[1])
                print(msg)
                logging.info(msg)
                if np.mod(counter, sample_interval) == 0:
                    samples = self.generator.predict(sample_inputs)
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                    save_images(samples, [manifold_h, manifold_w],
                        './{}/{}_train_{:02d}_{:04d}.png'.format(self.sample_dir, self.dataset_name, epoch, idx))

            # Save the checkpoint end of each epoch.
            self.save(epoch)
        # Export the Generator/R network reconstruction losses as a plot.
        plt.title('Generator/R network reconstruction losses')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(plot_epochs,plot_g_recon_losses)
        plt.savefig('{}_plot_g_recon_losses_multi.png'.format(self.dataset_name))

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.dataset_name,
            self.output_height, self.output_width)

    def save(self, step):
        """Helper method to save model weights.

        Arguments:
            step {[type]} -- [description]
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        model_name = '{}_ALOCC_Model_{}.h5'.format(self.dataset_name, step)
        self.adversarial_model.save_weights(os.path.join(self.checkpoint_dir, model_name))


if __name__ == '__main__':
    model = ALOCC_Model(dataset_name='cifar10')
    model.train(epochs=10, batch_size=128, sample_interval=500)
