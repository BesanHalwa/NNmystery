import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

def vgg16(classes=10, input_shape=(32,32,3)):

    img_input = (32,32,3)
    x = tf.keras.Sequential()

    # Block 1
    x.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(32,32,3)))
    x.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    x.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    x.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    x.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    x.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    x.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    x.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    x.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    x.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    x.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    x.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    x.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))


    x.add(tf.keras.layers.Flatten(name='flatten'))
    x.add(tf.keras.layers.Dense(512, activation='relu', name='fc1'))
    x.add(tf.keras.layers.Dense(512, activation='relu', name='fc2'))
    x.add(tf.keras.layers.Dense(classes, activation='softmax', name='predictions'))

    return x

def small(classes = 10, input_shape=(32,32,3)):

    x = tf.keras.models.Sequential()

    x.add(tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=input_shape, activation='relu'))
    x.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
    x.add(tf.keras.layers.MaxPooling2D())
    x.add(tf.keras.layers.Dropout(0.25))

    x.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    x.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
    x.add(tf.keras.layers.MaxPooling2D())
    x.add(tf.keras.layers.Dropout(0.25))

    x.add(tf.keras.layers.Flatten())
    x.add(tf.keras.layers.Dense(512, activation='relu'))
    x.add(tf.keras.layers.Dropout(0.5))
    x.add(tf.keras.layers.Dense(classes, activation='softmax'))

    return x


def tNet(classes = 10, input_shape=(32,32,3)):

  x = tf.keras.Sequential()

  #add layers to the model
  x.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
  #x.add(tf.keras.layers.MaxPooling2D(pool_size=2))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dropout(0.05))

  x.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
  #x.add(tf.keras.layers.MaxPooling2D(pool_size=2))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dropout(0.05))

  x.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
  #x.add(tf.keras.layers.MaxPooling2D(pool_size=2))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dropout(0.05))

  x.add(tf.keras.layers.Flatten())

  x.add(tf.keras.layers.Dense(512, activation='relu'))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dense(512, activation='relu'))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dense(256, activation='relu'))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dense(256, activation='relu'))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dense(128, activation='relu'))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dropout(0.05))

  x.add(tf.keras.layers.Dense(64, activation='relu'))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dense(32, activation='relu'))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dense(16, activation='relu'))
  x.add(tf.keras.layers.BatchNormalization())
  x.add(tf.keras.layers.Dropout(0.05))

  x.add(tf.keras.layers.Dense(classes, activation='softmax'))

  return x


def convNet(classes = 10, input_shape=(32,32,3)):

    x = tf.keras.Sequential()

    # Block 1
    x.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(32,32,3)))
    x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    x.add(tf.keras.layers.BatchNormalization())
    #x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    x.add(tf.keras.layers.Dropout(0.2))

    # Block 2
    x.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    x.add(tf.keras.layers.Dropout(0.2))

    # Block 3
    x.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    x.add(tf.keras.layers.Dropout(0.2))

    # Classification block
    x.add(tf.keras.layers.Flatten(name='flatten'))
    x.add(tf.keras.layers.Dropout(0.2))
    x.add(tf.keras.layers.Dense(40, activation='relu', name='fc1'))
    x.add(tf.keras.layers.Dense(40, activation='relu', name='fc2'))
    x.add(tf.keras.layers.Dense(classes, activation='softmax', name='predictions'))

    return x

def vanilaConvNet(classes = 10, input_shape=(32,32,3)):

    x = tf.keras.Sequential()

    # Block 1
    x.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
    #x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    #x.add(tf.keras.layers.BatchNormalization())
    #x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    #x.add(tf.keras.layers.Dropout(0.2))

    # Block 2
    x.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    #x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    #x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    #x.add(tf.keras.layers.Dropout(0.2))

    # Block 3
    x.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    #x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    #x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    #x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    #x.add(tf.keras.layers.Dropout(0.2))

    # Classification block
    x.add(tf.keras.layers.Flatten(name='flatten'))
    #x.add(tf.keras.layers.Dropout(0.2))
    x.add(tf.keras.layers.Dense(40, activation='relu', name='fc1'))
    x.add(tf.keras.layers.Dense(40, activation='relu', name='fc2'))
    x.add(tf.keras.layers.Dense(classes, activation='softmax', name='predictions'))

    return x
