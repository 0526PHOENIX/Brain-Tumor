"""
===================================================================================================================
Package
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dropout
from keras.layers import Add
from keras.layers import Conv2DTranspose
from keras.models import Model


"""
===================================================================================================================
Residual Block
"""
def residual_block(filters):

    def layer(img_input):

        x = BatchNormalization()(img_input)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size = (3, 3), padding = 'same', use_bias = False)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size = (3, 3), padding = 'same', use_bias = False)(x)

        x = Add()([x, img_input])

        return x
    
    return layer


"""
===================================================================================================================
Downsampling Block
"""
def down_block(filters):

    def layer(img_input):

        x = Conv2D(filters, kernel_size = (3, 3), strides = (2, 2), padding = 'same', use_bias = False)(img_input)

        return x
    
    return layer


"""
===================================================================================================================
Deconvolutional Block
"""
def up_block(filters):

    def layer(img_input):

        x = Conv2D(filters, kernel_size = (1, 1), padding = 'same', use_bias = False)(img_input)
        x = Conv2DTranspose(filters, kernel_size = (2, 2), strides = (2, 2), padding = 'same', use_bias = False)(x)

        return x
    
    return layer


"""
===================================================================================================================
Initial Block
"""
def initial_block(filters):

    def layer(img_input):

        x = Conv2D(filters, kernel_size = (3, 3), padding = 'same', use_bias = False)(img_input)
        x = Dropout(rate = 0.2)(x)

        return x 
    
    return layer


"""
===================================================================================================================
Final Block
"""
def final_block(filters):

    def layer(img_input):

        x = Conv2D(filters, kernel_size = (1, 1), padding = 'same', use_bias = False)(img_input)
        x = Activation('sigmoid', name = 'final')(x)

        return x 
    
    return layer


"""
===================================================================================================================
U-Net
Parameters: 9,953,088
"""
def unet(img_rows = 128, img_cols = 128, color_type = 1, num_class = 1):

    """
    ===============================================================================================================
    Filters
    """
    filters = [32, 64, 128, 256, 512]


    """
    ===============================================================================================================
    Encoder
    """
    img_input = Input(shape = (img_rows, img_cols, color_type))

    conv = initial_block(filters[0])(img_input)

    encode_1 = residual_block(filters[0])(conv)
    down_1 = down_block(filters[1])(encode_1)

    encode_2 = residual_block(filters[1])(down_1)
    down_2 = down_block(filters[2])(encode_2)

    encode_3 = residual_block(filters[2])(down_2)
    down_3 = down_block(filters[3])(encode_3)

    encode_4 = residual_block(filters[3])(down_3)
    down_4 = down_block(filters[4])(encode_4)

    encode_5 = residual_block(filters[4])(down_4)


    """
    ===============================================================================================================
    Decoder
    """
    up_4 = up_block(filters[3])(encode_5)
    decode_4 = Add()([up_4, encode_4])
    decode_4 = residual_block(filters[3])(decode_4)

    up_3 = up_block(filters[2])(decode_4)
    decode_3 = Add()([up_3, encode_3])
    decode_3 = residual_block(filters[2])(decode_3)
    
    up_2 = up_block(filters[1])(decode_3)
    decode_2 = Add()([up_2, encode_2])
    decode_2 = residual_block(filters[1])(decode_2)

    up_1 = up_block(filters[0])(decode_2)
    decode_1 = Add()([up_1, encode_1])
    decode_1 = residual_block(filters[0])(decode_1)

    final = final_block(filters = num_class)(decode_1)


    """
    ===============================================================================================================
    Model Shape
    """
    """
    # Format Setting
    space = "{:15}{:20}"

    # Encoder
    print(space.format('img_input:', str(img_input.shape)))
    print(space.format('conv:', str(conv.shape)))
    print(space.format('encode_1:', str(encode_1.shape)))
    print(space.format('down_1:', str(down_1.shape)))
    print(space.format('encode_2:', str(encode_2.shape)))
    print(space.format('down_2:', str(down_2.shape)))
    print(space.format('encode_3:', str(encode_3.shape)))
    print(space.format('down_3:', str(down_3.shape)))
    print(space.format('encode_4:', str(encode_4.shape)))
    print(space.format('down_4:', str(down_4.shape)))
    print(space.format('encode_5:', str(encode_5.shape)), '\n')

    # Decoder
    print(space.format('up_4:', str(up_4.shape)))
    print(space.format('decode_4:', str(decode_4.shape)))
    print(space.format('up_3:', str(up_3.shape)))
    print(space.format('decode_3:', str(decode_3.shape)))
    print(space.format('up_2:', str(up_2.shape)))
    print(space.format('decode_2:', str(decode_2.shape)))
    print(space.format('up_1:', str(up_1.shape)))
    print(space.format('decode_1:', str(decode_1.shape)))
    print(space.format('final:', str(final.shape)), '\n')
    """


    """
    ===============================================================================================================
    Model
    """
    model = Model(img_input, final)

    return model


"""
===================================================================================================================
Main Function
"""
if __name__ == '__main__':

    model = unet(img_rows = 128, img_cols = 128, color_type = 1, num_class = 1)
    print('Parameters:', model.count_params())