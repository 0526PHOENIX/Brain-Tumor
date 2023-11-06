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
def final_block(filters, stage = ''):

    def layer(img_input):

        x = Conv2D(filters, kernel_size = (1, 1), padding = 'same', use_bias = False)(img_input)
        x = Activation('sigmoid', name = 'final_' + stage)(x)

        return x 
    
    return layer


"""
===================================================================================================================
U-Net++
Parameters: 10,619,552
"""
def xnet(img_rows = 128, img_cols = 128, color_type = 1, num_class = 1):

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
    # Level 1
    up1_1 = up_block(filters[0])(encode_2)
    decode1_1 = Add()([up1_1, encode_1])
    decode1_1 = residual_block(filters[0])(decode1_1)

    final1 = final_block(filters = num_class, stage = '1')(decode1_1)

    # Level 2
    up2_2 = up_block(filters[1])(encode_3)
    decode2_2 = Add()([up2_2, encode_2])
    decode2_2 = residual_block(filters[1])(decode2_2)

    up2_1 = up_block(filters[0])(decode2_2)
    decode2_1 = Add()([up2_1, encode_2, decode1_1])
    decode2_1 = residual_block(filters[0])(decode2_1)

    final2 = final_block(filters = num_class, stage = '2')(decode2_1)

    # Level 3
    up3_3 = up_block(filters[2])(encode_4)
    decode3_3 = Add()([up3_3, encode_3])
    decode3_3 = residual_block(filters[2])(decode3_3)

    up3_2 = up_block(filters[1])(encode_3)
    decode3_2 = Add()([up3_2, encode_2, decode2_2])
    decode3_2 = residual_block(filters[1])(decode3_2)

    up3_1 = up_block(filters[0])(decode3_2)
    decode3_1 = Add()([up3_1, encode_2, decode1_1, decode2_1])
    decode3_1 = residual_block(filters[0])(decode3_1)

    final3 = final_block(filters = num_class, stage = '3')(decode3_1)

    # Level 4
    up4_4 = up_block(filters[3])(encode_5)
    decode4_4 = Add()([up4_4, encode_4])
    decode4_4 = residual_block(filters[3])(decode4_4)

    up4_3 = up_block(filters[2])(decode4_4)
    decode4_3 = Add()([up4_3, encode_3, decode3_3])
    decode4_3 = residual_block(filters[2])(decode4_3)
    
    up4_2 = up_block(filters[1])(decode4_3)
    decode4_2 = Add()([up4_2, encode_2, decode2_2, decode3_2])
    decode4_2 = residual_block(filters[1])(decode4_2)

    up4_1 = up_block(filters[0])(decode4_2)
    decode4_1 = Add()([up4_1, encode_1, decode1_1, decode2_1, decode3_1])
    decode4_1 = residual_block(filters[0])(decode4_1)

    final4 = final_block(filters = num_class, stage = '4')(decode4_1)


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
    # Level 1
    print(space.format('up1_1:', str(up1_1.shape)))
    print(space.format('decode1_1:', str(decode1_1.shape)))
    print(space.format('final1:', str(final1.shape)), '\n')

    # Level 2
    print(space.format('up2_2:', str(up2_2.shape)))
    print(space.format('decode2_2:', str(decode2_2.shape)))
    print(space.format('up2_1:', str(up2_1.shape)))
    print(space.format('decode2_1:', str(decode2_1.shape)))
    print(space.format('final2:', str(final2.shape)), '\n')

    # Level 3
    print(space.format('up3_3:', str(up3_3.shape)))
    print(space.format('decode3_3:', str(decode3_3.shape)))
    print(space.format('up3_2:', str(up3_2.shape)))
    print(space.format('decode3_2:', str(decode3_2.shape)))
    print(space.format('up3_1:', str(up3_1.shape)))
    print(space.format('decode3_1:', str(decode3_1.shape)))
    print(space.format('final3:', str(final3.shape)), '\n')

    # Level 4
    print(space.format('up4_4:', str(up4_4.shape)))
    print(space.format('decode4_4:', str(decode4_4.shape)))
    print(space.format('up4_3:', str(up4_3.shape)))
    print(space.format('decode4_3:', str(decode4_3.shape)))
    print(space.format('up4_2:', str(up4_2.shape)))
    print(space.format('decode4_2:', str(decode4_2.shape)))
    print(space.format('up4_1:', str(up4_1.shape)))
    print(space.format('decode4_1:', str(decode4_1.shape)))
    print(space.format('final4:', str(final4.shape)), '\n')
    """


    """
    ===============================================================================================================
    Model
    """
    model = Model(img_input, [final1, final2, final3, final4])

    return model


"""
===================================================================================================================
Main Function
"""
if __name__ == '__main__':

    model = xnet(img_rows = 128, img_cols = 128, color_type = 1, num_class = 1)
    print('Parameters:', model.count_params())