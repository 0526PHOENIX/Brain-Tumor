"""
===================================================================================================================
Package
"""
from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Add
from keras.models import Model

from keras.applications import DenseNet121, DenseNet169, DenseNet201


"""
===================================================================================================================
ConvRelu Block
"""
def ConvRelu(filters):

    def layer(img_input):

        x = BatchNormalization()(img_input)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size = (3, 3), padding = 'same', use_bias = False)(x)

        x = Add()([x, img_input])

        return x
    
    return layer


"""
===================================================================================================================
Deconvolutional Block
"""
def up_block(filters, skip = None):

    def layer(img_input):

        x = Conv2D(filters, kernel_size = (1, 1), padding = 'same', use_bias = False)(img_input)
        x = Conv2DTranspose(filters, kernel_size = (2, 2), strides = (2, 2), padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if skip is not None:
            x = Add()([x, skip])

        x = ConvRelu(filters)(x)

        return x
    
    return layer


"""
===================================================================================================================
Backbone
"""
backbones = {
                "densenet121": DenseNet121,
                "densenet169": DenseNet169,
                "densenet201": DenseNet201,
            }

def get_backbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs)


"""
===================================================================================================================
Build Model
"""
def build_unet(backbone, classes, skip_connection_layers,
               decoder_filters = (512, 256, 128, 64, 32),
               n_upsample_blocks = 5,
               activation = 'sigmoid'):

    img_input = backbone.input
    x = backbone.output

    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l for l in skip_connection_layers])

    for i in range(n_upsample_blocks):

        skip_connection = None

        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        x = up_block(decoder_filters[i], skip = skip_connection)(x)

    x = Conv2D(classes, kernel_size = (3, 3), padding = 'same', use_bias = False)(x)
    x = Activation(activation)(x)

    model = Model(img_input, x)

    return model


"""
===================================================================================================================
Set Model
Parameters: 12,283,488
"""
DEFAULT_SKIP_CONNECTIONS =  {
                                'densenet121': (311, 139, 51, 4),
                                'densenet169': (367, 139, 51, 4),
                                'densenet201': (479, 139, 51, 4),
                            }


def Unet(backbone_name = 'densenet121',
         input_shape = (None, None, 3),
         input_tensor = None,
         encoder_weights = 'imagenet',
         decoder_filters = (512, 256, 128, 64, 32),
         n_upsample_blocks = 5,
         classes = 1,
         activation = 'sigmoid'):


    backbone = get_backbone(backbone_name,
                            input_shape = input_shape,
                            input_tensor = input_tensor,
                            weights = encoder_weights,
                            include_top = False)


    skip_connections = DEFAULT_SKIP_CONNECTIONS[backbone_name]

    model = build_unet(backbone, classes, skip_connections,
                       decoder_filters = decoder_filters,
                       n_upsample_blocks = n_upsample_blocks,
                       activation = activation)


    return model


"""
===================================================================================================================
Main Function
"""
if __name__ == '__main__':

    model = Unet(backbone_name = 'densenet121',
                 encoder_weights = 'imagenet',
                 classes = 1,
                 activation = 'sigmoid')
    
    print('Parameters:', model.count_params())