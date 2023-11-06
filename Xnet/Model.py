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
        x = Conv2DTranspose(filters, kernel_size = (4, 4), strides = (2, 2), padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if (type(skip) != list and skip is not None) or (type(skip) == list and None not in skip):
            if type(skip) is list:
                merge_list = []
                merge_list.append(x)

                for l in skip:
                    merge_list.append(l)

                x = Add()(merge_list)
            else:
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
def build_xnet(backbone, classes, skip_connection_layers,
               decoder_filters = (512, 256, 128, 64, 32),
               n_upsample_blocks = 5,
               activation = 'sigmoid'):

    img_input = backbone.input

    downsampling_layers = skip_connection_layers

    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l for l in skip_connection_layers])
    skip_layers_list = [backbone.layers[skip_connection_idx[i]].output for i in range(len(skip_connection_idx))]
    
    downsampling_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l for l in downsampling_layers])
    downsampling_list = [backbone.layers[downsampling_idx[i]].output for i in range(len(downsampling_idx))]
    
    downterm = [None] * (n_upsample_blocks + 1)
    
    for i in range(len(downsampling_idx)):
        downterm[n_upsample_blocks - i - 1] = downsampling_list[i]
    downterm[-1] = backbone.output

    interm = [None] * (n_upsample_blocks + 1) * (n_upsample_blocks + 1)
    for i in range(len(skip_connection_idx)):
        interm[-i * (n_upsample_blocks + 1) + (n_upsample_blocks + 1) * (n_upsample_blocks - 1)] = skip_layers_list[i]
    interm[(n_upsample_blocks + 1) * n_upsample_blocks] = backbone.output

    for j in range(n_upsample_blocks):
        for i in range(n_upsample_blocks - j):
            if j == 0:
                if downterm[i + 1] is not None:
                    interm[(n_upsample_blocks + 1) * i + j + 1] = up_block(decoder_filters[n_upsample_blocks - i - 1],
                                      skip = interm[(n_upsample_blocks + 1) * i + j])(downterm[i + 1])
                else:
                    interm[(n_upsample_blocks + 1) * i + j + 1] = None
            else:
                interm[(n_upsample_blocks+1) * i + j + 1] = up_block(decoder_filters[n_upsample_blocks - i - 1],
        skip = interm[(n_upsample_blocks + 1) * i : (n_upsample_blocks + 1) * i + j + 1])(interm[(n_upsample_blocks + 1) * (i + 1) + j])

    x = Conv2D(classes, kernel_size = (3, 3), padding = 'same', use_bias = False)(interm[n_upsample_blocks])
    x = Activation(activation)(x)

    model = Model(img_input, x)

    return model


"""
===================================================================================================================
Set Model
Parameters: 19,465,312
"""
DEFAULT_SKIP_CONNECTIONS =  {
                                'densenet121': (311, 139, 51, 4),
                                'densenet169': (367, 139, 51, 4),
                                'densenet201': (479, 139, 51, 4),
                            }


def Xnet(backbone_name = 'densenet121',
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

    model = build_xnet(backbone, classes, skip_connections,
                       decoder_filters = decoder_filters,
                       activation = activation,
                       n_upsample_blocks = n_upsample_blocks,)

    return model


"""
===================================================================================================================
Main Function
"""
if __name__ == '__main__':

    model = Xnet(backbone_name = 'densenet121',
                 encoder_weights = 'imagenet',
                 classes = 1,
                 activation = 'sigmoid')
    
    print('Parameters:', model.count_params())