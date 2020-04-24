import os
import tensorflow as tf
import numpy as np

#import helper
### HELPER ###
import math
import os
import hashlib
from urllib.request import urlretrieve
import zipfile
import gzip
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def _read32(bytestream):
    """
    Read 32-bit integer from bytesteam
    :param bytestream: A bytestream
    :return: 32-bit integer
    """
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)


def _ungzip(save_path, extract_path, database_name, _):
    """
    Unzip a gzip file and extract it to extract_path
    :param save_path: The path of the gzip files
    :param extract_path: The location to extract the data to
    :param database_name: Name of database
    :param _: HACK - Used to have to same interface as _unzip
    """
    # Get data from save_path
    with open(save_path, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number {} in file: {}'.format(magic, f.name))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)

    # Save data to extract_path
    for image_i, image in enumerate(
            tqdm(data, unit='File', unit_scale=True, miniters=1, desc='Extracting {}'.format(database_name))):
        Image.fromarray(image, 'L').save(os.path.join(extract_path, 'image_{}.jpg'.format(image_i)))


def get_image(image_path, width, height, mode):
    """
    Read image from image_path
    :param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param mode: Mode of image
    :return: Image data
    """
    image = Image.open(image_path)

    if image.size != (width, height):  # HACK - Check if image is from the CELEBA dataset
        # Remove most pixels that aren't part of a face
        face_width = face_height = 108
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert(mode))


def get_batch(image_files, width, height, mode):
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch


def images_square_grid(images, mode):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im


def download_extract(database_name, data_path):
    """
    Download and extract database
    :param database_name: Database name
    """
    DATASET_CELEBA_NAME = 'celeba'
    DATASET_MNIST_NAME = 'mnist'

    if database_name == DATASET_CELEBA_NAME:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        extract_path = os.path.join(data_path, 'img_align_celeba')
        save_path = os.path.join(data_path, 'celeba.zip')
        extract_fn = _unzip
    elif database_name == DATASET_MNIST_NAME:
        url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        hash_code = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
        extract_path = os.path.join(data_path, 'mnist')
        save_path = os.path.join(data_path, 'train-images-idx3-ubyte.gz')
        extract_fn = _ungzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)

    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
        '{} file is corrupted.  Remove the file and try again.'.format(save_path)

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, database_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)


class Dataset(object):
    """
    Dataset
    """
    def __init__(self, dataset_name, data_files):
        """
        Initalize the class
        :param dataset_name: Database name
        :param data_files: List of files in the database
        """
        DATASET_CELEBA_NAME = 'celeba'
        DATASET_MNIST_NAME = 'mnist'
        IMAGE_WIDTH = 28
        IMAGE_HEIGHT = 28

        if dataset_name == DATASET_CELEBA_NAME:
            self.image_mode = 'RGB'
            image_channels = 3

        elif dataset_name == DATASET_MNIST_NAME:
            self.image_mode = 'L'
            image_channels = 1

        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels

    def get_batches(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch Size
        :return: Batches of data
        """
        IMAGE_MAX_VALUE = 255

        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3],
                self.image_mode)

            current_index += batch_size

            yield data_batch / IMAGE_MAX_VALUE - 0.5


class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
### HELPER ###

from glob import glob
import pickle as pkl
import scipy.misc

import time

import cv2
import matplotlib.pyplot as plt

# If you want to make a model from scratch, set from_checkpoint to False
# If you don't want your training to take forever, set do_preprocess to True
do_preprocess = True
from_checkpoint = False

# Declare your directory variables here
data_dir = './laxtraining'
data_resized_dir = './resized_laxtraining'

# Import the data and preprocess it for size
if do_preprocess == True:
    os.mkdir(data_resized_dir)

    for each in os.listdir(data_dir):
        image = cv2.imread(os.path.join(data_dir, each))
        image = cv2.resize(image, (128, 128))
        cv2.imwrite(os.path.join(data_resized_dir, each), image)

# Declare a function to get images and convert them
def get_image(image_path, width, height, mode):
    image = Image.Open(image.path)
    return np.array(image.convert(mode))

def get_batch(image_files, width, height, mode):
    data_batch = np.array([get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in four dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch

# Explore the data
show_n_images = 25
mnist_images = get_batch(glob(os.path.join(data_resized_dir, '*.jpg'))[:show_n_images], 64, 64, 'RGB')
plt.imshow(images_square_grid(mnist_images, 'RGB'))

# Check TensorFlow version and access to Nvidia GPU
from distutils.version import LooseVersion
import warnings

### Check the TensorFlow version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer. You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

### Check for the presence of a GPU for graphics hardware acceleration
### It isn't impossible to run without a GPU, but it will take significantly longer
if not tf.test.gpu_device_name():
    warnings.warn('No compatible GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU device: {}'.format(tf.tetst.gpu_device_name()))

# Create TensorFlow placeholders for the Neural Network
### Real input images placeholder is real_dim
### Z input placeholder is z_dim
### Learning rate placeholders are G and D
### Placeholders are returned in a tuple

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learning_rate_G = tf.placeholder(tf.float32, name = 'learning_rate_G')
    learning_rate_D = tf.placeholder(tf.float32, name = 'learning_rate_D')

    return inputs_real, inputs_z, learning_rate_G, learning_rate_D

# Generator
def generator(z, output_channel_dim, is_train=True):
    """
    Build the generator network

    Arguments
    ---------
    z : Input tensor for the generator
    output_channel_dim : Shape of the generator output
    n_units : Number of units in hidden layer
    reuse : Reuse the variables with tf.variable_scope
    alpha : Leak parameter for leaky ReLU

    Returns
    -------
    out:
    """
    with tf.variable_scope("generator", reuse= not is_train):
        # First FC layer --> 8x8x1024
        fc1 = tf.layers.dense(z, 8*8*1024)

        # Reshape fc1
        fc1 = tf.reshape(fc1, (-1, 8, 8, 1024))

        # Leaky ReLU
        fc1 = tf.nn.leaky_relu(fc1, alpha=alpha)

        # Transposed conv 1 --> BatchNorm --> LeakyReLU
        # 8x8x1024 --> 16x16x512
        trans_conv = tf.layers.conv2d_transpose(inputs = fc1,
            filters = 512,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "SAME",
            kernel_initializer = tf.truncated_normal_initializer(sddev=0.02),
            name="trans_conv1")

        batch_trans_conv1 = tf.layers.batch_normalization(inputs = trans_conv1, training = is_train, epsilon = 1e-5, name = "batch_trans_conv1")

        trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv1, alpha=alpha, name="trans_conv1_out")

        # Transposed conv 2 --> BatchNorm --> LeakyReLU
        # 16x16x512 --> 32x32x256
        trans_conv2 = tf.layers.conv2d_transpose(inputs = trans_conv1_out,
            filters = 256,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "SAME",
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
            name = "trans_conv2")

        batch_trans_conv2 = tf.layers.batch_normalization(inputs = trans_conv2, training = is_train, epsilon = 1e-5, name = "batch_trans_conv2")
        trans_conv2_out = tf.nn.leaky_relu(batch_trans_conv2, alpha=alpha, name="trans_conv2_out")

        # Transposed conv 3 --> BatchNorm --> LeakyReLU
        # 32x32x256 --> 64x64x128
        trans_conv3 = tf.layers.conv3d_transpose(inputs = trans_conv2_out,
            filters = 128,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "SAME",
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
            name="trans_conv3")

        batch_trans_conv3 = tf.layers.batch_normalization(inputs = trans_conv3, training = is_train, epsilon = 1e-5, name = "batch_trans_conv3")
        trans_conv3_out = tf.nn.leaky_relu(batch_trans_conv3, alpha=alpha, name="trans_conv3_out")

        # Transposed conv 4 --> BatchNorm --> LeakyReLU
        # 64x64x128 --> 128x128x64
        trans_conv4 = tf.layers.conv2d_transpose(inputs = trans_conv3_out,
            filters = 64,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "SAME",
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
            name = "trans_conv4")

        batch_trans_conv4 = tf.layers.batch_normalization(inputs = trans_conv4, training = is_train, epsilon = 1e-5, name = "batch_trans_conv4")
        trans_conv4_out = tf.nn.leaky_relu(batch_trans_conv4, alpha=alpha, name="trans_conv4_out")

        # Transposed conv5 --> tanh
        # 128x128x64 --> 128x128x3
        logits = tf.layers.conv2d_transpose(inputs = trans_conv4_out,
            filters = 3,
            kernel_size = [5,5],
            strides = [1,1],
            padding = "SAME",
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
            name = "logits")

        out = tf.tanh(logits, name = "out")

        return out

def discriminator(x, is_reuse=False, alpha = 0.2):
    """
    Build the discriminator network.

        Arguments
        ---------
        x : Input tensor for the discriminator
        n_units: Number of units in hidden layer
        reuse : Reuse the variables with tf.variable_scope
        alpha : leak parameter for leaky ReLU

        Returns
        -------
        out, logits:
    """

    with tf.variable_scope("discriminator", reuse = is_reuse):

        # Input layer 128*128*3 --> 64x64x64
        # Conv --> BatchNorm --> LeakyReLU
        conv1 = tf.layers.conv2d(inputs = x,
            filters = 64,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "SAME",
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name='conv1')

        batch_norm1 = tf.layers.batch_normalization(conv1,
            training = True,
            epsilon = 1e-5,
            name = 'batch_norm1')

        conv1_out = tf.nn.leaky_relu(batch_norm1, alpha=alpha, name="conv1_out")


        # 64x64x64--> 32x32x128
        # Conv --> BatchNorm --> LeakyReLU
        conv2 = tf.layers.conv2d(inputs = conv1_out,
            filters = 128,
            kernel_size = [5, 5],
            strides = [2, 2],
            padding = "SAME",
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name='conv2')

        batch_norm2 = tf.layers.batch_normalization(conv2,
           training = True,
           epsilon = 1e-5,
           name = 'batch_norm2')

        conv2_out = tf.nn.leaky_relu(batch_norm2, alpha=alpha, name="conv2_out")



        # 32x32x128 --> 16x16x256
        # Conv --> BatchNorm --> LeakyReLU
        conv3 = tf.layers.conv2d(inputs = conv2_out,
            filters = 256,
            kernel_size = [5, 5],
            strides = [2, 2],
            padding = "SAME",
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name='conv3')

        batch_norm3 = tf.layers.batch_normalization(conv3,
           training = True,
           epsilon = 1e-5,
           name = 'batch_norm3')

        conv3_out = tf.nn.leaky_relu(batch_norm3, alpha=alpha, name="conv3_out")



        # 16x16x256 --> 16x16x512
        # Conv --> BatchNorm --> LeakyReLU
        conv4 = tf.layers.conv2d(inputs = conv3_out,
            filters = 512,
            kernel_size = [5, 5],
            strides = [1, 1],
            padding = "SAME",
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name='conv4')

        batch_norm4 = tf.layers.batch_normalization(conv4,
            training = True,
            epsilon = 1e-5,
            name = 'batch_norm4')

        conv4_out = tf.nn.leaky_relu(batch_norm4, alpha=alpha, name="conv4_out")



        # 16x16x512 --> 8x8x1024
        # Conv --> BatchNorm --> LeakyReLU
        conv5 = tf.layers.conv2d(inputs = conv4_out,
            filters = 1024,
            kernel_size = [5, 5],
            strides = [2, 2],
            padding = "SAME",
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name='conv5')

        batch_norm5 = tf.layers.batch_normalization(conv5,
            training = True,
            epsilon = 1e-5,
            name = 'batch_norm5')

        conv5_out = tf.nn.leaky_relu(batch_norm5, alpha=alpha, name="conv5_out")


        # Flatten it
        flatten = tf.reshape(conv5_out, (-1, 8*8*1024))

        # Logits
        logits = tf.layers.dense(inputs = flatten,
            units = 1,
            activation = None)


        out = tf.sigmoid(logits)

        return out, logits

def model_loss(input_real, input_z, output_channel_dim, alpha):
    """
    Get the loss for the discriminator and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param out_channel_dim: The number of channels in the output image
    :return: A tuple of (discriminator loss, generator loss)
    """
    # Generator network here
    g_model = generator(input_z, output_channel_dim)
    # g_model is the generator output

    # Discriminator network here
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model,is_reuse=True, alpha=alpha)

    # Calculate losses
    d_loss_real = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                          labels=tf.ones_like(d_model_real)))
    d_loss_fake = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                          labels=tf.zeros_like(d_model_fake)))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                     labels=tf.ones_like(d_model_fake)))

    return d_loss, g_loss

def model_optimizers(d_loss, g_loss, lr_D, lr_G, beta1):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get the trainable_variables, split into G and D parts
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Generator update
    gen_updates = [op for op in update_ops if op.name.startswith('generator')]

    # Optimizers
    with tf.control_dependencies(gen_updates):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=lr_D, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=lr_G, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode, image_path, save, show):
    """
    Show example output for the generator
    :param sess: TensorFlow session
    :param n_images: Number of Images to display
    :param input_z: Input Z Tensor
    :param out_channel_dim: The number of channels in the output image
    :param image_mode: The mode to use for images ("RGB" or "L")
    :param image_path: Path to save the image
    """
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = images_square_grid(samples, image_mode)

    if save == True:
        # Save image
        images_grid.save(image_path, 'JPEG')

    if show == True:
        plt.imshow(images_grid, cmap=cmap)
        plt.show()

def train(epoch_count, batch_size, z_dim, learning_rate_D, learning_rate_G, beta1, get_batches, data_shape, data_image_mode, alpha):
    """
    Train the GAN
    :param epoch_count: Number of epochs
    :param batch_size: Batch Size
    :param z_dim: Z dimension
    :param learning_rate: Learning Rate
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :param get_batches: Function to get batches
    :param data_shape: Shape of the data
    :param data_image_mode: The image mode to use for images ("RGB" or "L")
    """
    # Create our input placeholders
    input_images, input_z, lr_G, lr_D = model_inputs(data_shape[1:], z_dim)

    # Losses
    d_loss, g_loss = model_loss(input_images, input_z, data_shape[3], alpha)

    # Optimizers
    d_opt, g_opt = model_optimizers(d_loss, g_loss, lr_D, lr_G, beta1)

    i = 0

    version = "firstTrain"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Saver
        saver = tf.train.Saver()

        num_epoch = 0

        if from_checkpoint == True:
            saver.restore(sess, "./models/model.ckpt")

            show_generator_output(sess, 4, input_z, data_shape[3], data_image_mode, image_path, True, False)

        else:
            for epoch_i in range(epoch_count):
                num_epoch += 1

                if num_epoch % 5 == 0:

                    # Save model every 5 epochs
                    #if not os.path.exists("models/" + version):
                    #    os.makedirs("models/" + version)
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model saved")

                for batch_images in get_batches(batch_size):
                    # Random noise
                    batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                    i += 1

                    # Run optimizers
                    _ = sess.run(d_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_D: learning_rate_D})
                    _ = sess.run(g_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_G: learning_rate_G})

                    if i % 10 == 0:
                        train_loss_d = d_loss.eval({input_z: batch_z, input_images: batch_images})
                        train_loss_g = g_loss.eval({input_z: batch_z})

                        # Save it
                        image_name = str(i) + ".jpg"
                        image_path = "./images/" + image_name
                        show_generator_output(sess, 4, input_z, data_shape[3], data_image_mode, image_path, True, False)

                    # Print every 5 epochs (for stability overwize the jupyter notebook will bug)
                    if i % 1500 == 0:

                        image_name = str(i) + ".jpg"
                        image_path = "./images/" + image_name
                        print("Epoch {}/{}...".format(epoch_i+1, epochs),
                              "Discriminator Loss: {:.4f}...".format(train_loss_d),
                              "Generator Loss: {:.4f}".format(train_loss_g))
                        show_generator_output(sess, 4, input_z, data_shape[3], data_image_mode, image_path, False, True)

    return losses, samples

# GANs are very sensitive to hyperparameters! Be careful!
# In general, discriminator loss should be around 0.3, meaning that it correctly classifies images as real or fake about half of the time
# Size input image for discriminator
real_size = (128, 128, 3)

# Size of latent vector to generator
z_dim = 100
learning_rate_D = .00005
learning_rate_G = 2e-4
batch_size = 64
epochs = 215
alpha = 0.2
beta1 = 0.5

# Create the network
#model = DGAN(real_size, z_size, learning_rate, alpha, beta1)

# Load the data and train the network here
dataset = Dataset(glob(os.path.join(data_resized_dir, '*.jpg')))

with tf.Graph().as_default():
    losses, samples = train(epochs, batch_size, z_dim, learning_rate_D, learning_rate_G, beta1, dataset.get_batches, dataset.shape, dataset.image_mode, alpha)

# Determine the training losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label = 'Discriminator', alpha = 0.5)
plt.plot(losses.T[1], label = 'Generator', alpha = 0.5)
plt.title("Training Losses")
plt.suptitle("At alpha = 0.5")
plt.legend()
