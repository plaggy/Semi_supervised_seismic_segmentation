from typing import Optional, Union, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.initializers import TruncatedNormal

tf.keras.backend.set_floatx('float64')


# some basic model used for testing, didn't try it at later stages
class BaseClassifier(tf.keras.Model):
    def __init__(self, n_classes, n_channels):
        super(BaseClassifier, self).__init__()
        self.c1 = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', input_shape=(None, None, n_channels))
        self.bn1 = layers.BatchNormalization()
        self.lr1 = layers.LeakyReLU()

        self.c2 = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')
        self.bn2 = layers.BatchNormalization()
        self.lr2 = layers.LeakyReLU()
        self.do2 = layers.Dropout(0.2)

        self.c3 = layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same')
        self.bn3 = layers.BatchNormalization()
        self.lr3 = layers.LeakyReLU()

        self.c4 = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same')
        self.bn4 = layers.BatchNormalization()
        self.lr4 = layers.LeakyReLU()
        self.do4 = layers.Dropout(0.2)

        self.c5 = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same')
        self.bn5 = layers.BatchNormalization()
        self.lr5 = layers.LeakyReLU()

        self.c_final = layers.Conv2D(n_classes, (1, 1), strides=(1, 1), padding='same')
        self.activ_final = layers.Softmax()

    def call(self, x, training=True, mask=None):
        x = self.lr1(self.bn1(self.c1(x)))
        x = self.do2(self.lr2(self.bn2(self.c2(x))))
        x = self.lr3(self.bn3(self.c3(x)))
        x = self.do4(self.lr4(self.bn4(self.c4(x))))
        x = self.lr5(self.bn5(self.c5(x)))
        x = self.c_final(x)
        y = self.activ_final(x)

        return y


class ConvBlock(layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.dropout_rate=dropout_rate
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx, filters_root)
        self.conv2d_1 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer='he_normal',
                                      padding=padding)
        self.dropout_1 = layers.Dropout(rate=dropout_rate)
        self.activation_1 = layers.Activation(activation)

        self.conv2d_2 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer='he_normal',
                                      padding=padding)
        self.dropout_2 = layers.Dropout(rate=dropout_rate)
        self.activation_2 = layers.Activation(activation)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.conv2d_1(x)

        x = self.dropout_1(x)

        x = self.activation_1(x)
        x = self.conv2d_2(x)

        x = self.dropout_2(x)

        x = self.activation_2(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                    activation=self.activation,
                    **super(ConvBlock, self).get_config(),
                    )


class UpconvBlock(layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx + 1, filters_root)
        self.upconv = layers.Conv2DTranspose(filters // 2,
                                             kernel_size=(pool_size, pool_size),
                                             kernel_initializer='he_normal',
                                             strides=pool_size, padding=padding)

        self.activation_1 = layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.upconv(x)
        x = self.activation_1(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    pool_size=self.pool_size,
                    padding=self.padding,
                    activation=self.activation,
                    **super(UpconvBlock, self).get_config())


class CropConcatBlock(layers.Layer):

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:,
                                        height_diff: (x1_shape[1] - height_diff),
                                        width_diff: (x1_shape[2] - width_diff),
                                        :]

        x = tf.concat([down_layer_cropped, x], axis=-1)
        return x

"""
standard Unet implementation
the source: https://github.com/jakeret/unet
"""
def build_unet(nx: Optional[int] = None,
               ny: Optional[int] = None,
               channels: int = 1,
               num_classes: int = 2,
               layer_depth: int = 5,
               filters_root: int = 64,
               kernel_size: int = 3,
               pool_size: int = 2,
               dropout_rate: float = 0.2,
               padding:str="same",
               activation:Union[str, Callable]="relu") -> Model:
    """
    Constructs a U-Net model

    :param nx: (Optional) image size on x-axis
    :param ny: (Optional) image size on y-axis
    :param channels: number of channels of the input tensors
    :param num_classes: number of classes
    :param layer_depth: total depth of unet
    :param filters_root: number of filters in top unet layer
    :param kernel_size: size of convolutional layers
    :param pool_size: size of maxplool layers
    :param dropout_rate: rate of dropout
    :param padding: padding to be used in convolutions
    :param activation: activation to be used

    :return: A TF Keras model
    """

    inputs = Input(shape=(nx, ny, channels), name="inputs")

    x = inputs
    contracting_layers = {}

    conv_params = dict(filters_root=filters_root,
                       kernel_size=kernel_size,
                       dropout_rate=dropout_rate,
                       padding=padding,
                       activation=activation)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = layers.MaxPooling2D((pool_size, pool_size))(x)

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = UpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation)(x)
        #x = CropConcatBlock()(x, contracting_layers[layer_idx])
        x = tf.concat([x, contracting_layers[layer_idx]], axis=-1)
        if layer_depth - layer_idx <= 2:
            x = layers.Dropout(dropout_rate)(x)
        x = ConvBlock(layer_idx, **conv_params)(x)

    x = layers.Conv2D(filters=num_classes, kernel_size=1, padding="same", name='features')(x)
    outputs = layers.Activation("softmax", name="outputs")(x)
    model = Model(inputs, outputs, name="unet")

    return model


"""
The same good old Unet but with an additional conv layer
To get a different representation of input examples to identify OOD samples
"""
def build_modified_unet(nx: Optional[int] = None,
                        ny: Optional[int] = None,
                        channels: int = 1,
                        num_classes: int = 2,
                        layer_depth: int = 5,
                        filters_root: int = 64,
                        kernel_size: int = 3,
                        pool_size: int = 2,
                        dropout_rate: float = 0.2,
                        padding:str="same",
                        activation:Union[str, Callable]="relu") -> Model:

    inputs = Input(shape=(nx, ny, channels), name="inputs")

    x = inputs
    contracting_layers = {}

    conv_params = dict(filters_root=filters_root,
                       kernel_size=kernel_size,
                       dropout_rate=dropout_rate,
                       padding=padding,
                       activation=activation)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = layers.MaxPooling2D((pool_size, pool_size))(x)

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = UpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation)(x)
        # x = CropConcatBlock()(x, contracting_layers[layer_idx])
        x = tf.concat([x, contracting_layers[layer_idx]], axis=-1)
        if layer_depth - layer_idx <= 2:
            x = layers.Dropout(dropout_rate)(x)
        x = ConvBlock(layer_idx, **conv_params)(x)

    # this is an additional layer to extract features to identify OOD samples
    x = layers.Conv2D(filters=7, kernel_size=1, padding="same", activation='relu', name='features')(x)

    x = layers.Conv2D(filters=num_classes, kernel_size=1, padding="same", name='final')(x)
    outputs = layers.Activation("softmax", name="outputs")(x)
    model = Model(inputs, outputs, name="unet")

    return model


"""
Unet with multiple heads instead of a single softmax, each head is represented by
a convolutional layer outputting logits and a sigmoid
The number of heads is equal to the number of classes, each learns features for a specific class
The modification is to identify OOD samples based on probability values
"""
def build_sigmoid_unet(nx: Optional[int] = None,
                       ny: Optional[int] = None,
                       channels: int = 1,
                       num_classes: int = 2,
                       layer_depth: int = 5,
                       filters_root: int = 64,
                       kernel_size: int = 3,
                       pool_size: int = 2,
                       dropout_rate: float = 0.2,
                       padding:str="same",
                       activation:Union[str, Callable]="relu") -> Model:

    inputs = Input(shape=(nx, ny, channels), name="inputs")

    x = inputs
    contracting_layers = {}

    conv_params = dict(filters_root=filters_root,
                       kernel_size=kernel_size,
                       dropout_rate=dropout_rate,
                       padding=padding,
                       activation=activation)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = layers.MaxPooling2D((pool_size, pool_size))(x)

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = UpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation)(x)

        x = tf.concat([x, contracting_layers[layer_idx]], axis=-1)
        if layer_depth - layer_idx <= 2:
            x = layers.Dropout(dropout_rate)(x)
        if layer_idx == 0:
            final_block = ConvBlock(layer_idx, **conv_params, name='before_head')(x)
        else:
            x = ConvBlock(layer_idx, **conv_params)(x)

    outputs = []
    for h in range(num_classes):
        x = layers.Conv2D(filters=1, kernel_size=1, padding="same")(final_block)
        output = layers.Activation("sigmoid")(x)
        outputs.append(output)
    model = Model(inputs, layers.Concatenate()(outputs), name="unet")

    return model


def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return TruncatedNormal(stddev=stddev)