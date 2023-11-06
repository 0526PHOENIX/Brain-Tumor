"""
===================================================================================================================
Package
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv3D, BatchNormalization, Dropout
from keras.layers import concatenate, Add
from keras.layers import Conv3DTranspose, UpSampling3D
from keras.models import Model


"""
===================================================================================================================
Residual Block
"""
def residual_block(filters):

    def layer(img_input):

        x = BatchNormalization()(img_input)
        x = Activation('relu')(x)
        x = Conv3D(filters, kernel_size = (3, 3, 3), padding = 'same', use_bias = False)(x)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(filters, kernel_size = (3, 3, 3), padding = 'same', use_bias = False)(x)

        x = Add()([x, img_input])

        return x
    
    return layer


"""
===================================================================================================================
Downsampling Block
"""
def down_block(filters):

    def layer(img_input):

        x = Conv3D(filters, kernel_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same', use_bias = False)(img_input)

        return x
    
    return layer


"""
===================================================================================================================
Deconvolutional Block
"""
def up_block_1(filters):

    def layer(img_input):

        x = Conv3D(filters, kernel_size = (1, 1, 1), padding = 'same', use_bias = False)(img_input)
        x = Conv3DTranspose(filters, kernel_size = (2, 2, 2), strides = (2, 2, 2), padding = 'same', use_bias = False)(x)

        return x
    
    return layer


"""
===================================================================================================================
Bilinear Interpolation Block
"""
def up_block_2(filters):

    def layer(img_input):

        x = Conv3D(filters, kernel_size = (1, 1, 1), padding = 'same', use_bias = False)(img_input)
        x = UpSampling3D(size = (2, 2, 2))(x)

        return x
    
    return layer


"""
===================================================================================================================
Initial Block
"""
def initial_block(filters):

    def layer(img_input):

        x = Conv3D(filters, kernel_size = (3, 3, 3), padding = 'same', use_bias = False)(img_input)
        x = Dropout(rate = 0.2)(x)

        return x 
    
    return layer


"""
===================================================================================================================
Final Block
"""
def final_block(filters, stage):

    def layer(img_input):

        x = Conv3D(filters, kernel_size = (1, 1, 1), padding = 'same', use_bias = False)(img_input)
        x = Activation('sigmoid', name = 'final_' + stage)(x)

        return x 
    
    return layer


"""
===================================================================================================================
Cascaded U-Net
Parameters: 24,975,440
"""
def cascaded_unet(img_size = 128, color_type = 4, num_class = 1):

    """
    ===============================================================================================================
    Filters
    """
    filters_1 = [16, 32, 64, 128]
    filters_2 = [32, 64, 128, 256]


    """
    ===============================================================================================================
    U-Net 1 Encoder
    """
    img_input_1 = Input(shape = (img_size, img_size, img_size, color_type))
    
    conv1 = initial_block(filters_1[0])(img_input_1)

    encode1_1 = residual_block(filters_1[0])(conv1)
    down1_1 = down_block(filters_1[1])(encode1_1)

    encode1_2 = residual_block(filters_1[1])(down1_1)
    encode1_2 = residual_block(filters_1[1])(encode1_2)
    down1_2 = down_block(filters_1[2])(encode1_2)

    encode1_3 = residual_block(filters_1[2])(down1_2)
    encode1_3 = residual_block(filters_1[2])(encode1_3)
    down1_3 = down_block(filters_1[3])(encode1_3)

    encode1_4 = residual_block(filters_1[3])(down1_3)
    encode1_4 = residual_block(filters_1[3])(encode1_4)
    encode1_4 = residual_block(filters_1[3])(encode1_4)
    encode1_4 = residual_block(filters_1[3])(encode1_4)


    """
    ===============================================================================================================
    U-Net 1 Decoder
    """
    up1_3 = up_block_1(filters_1[2])(encode1_4)
    decode1_3 = Add()([up1_3, encode1_3])
    decode1_3 = residual_block(filters_1[2])(decode1_3)
    
    up1_2 = up_block_1(filters_1[1])(decode1_3)
    decode1_2 = Add()([up1_2, encode1_2])
    decode1_2 = residual_block(filters_1[1])(decode1_2)

    up1_1 = up_block_1(filters_1[0])(decode1_2)
    decode1_1 = Add()([up1_1, encode1_1])
    decode1_1 = residual_block(filters_1[0])(decode1_1)

    final1 = final_block(filters = num_class, stage = '1')(decode1_1)


    """
    ===============================================================================================================
    U_Net 2 Encoder
    """
    img_input_2 = concatenate([final1, img_input_1], axis = -1)

    conv2 = initial_block(filters_2[0])(img_input_2)
    
    encode2_1 = residual_block(filters_2[0])(conv2)
    down2_1 = down_block(filters_2[1])(encode2_1)
    
    encode2_2 = residual_block(filters_2[1])(down2_1)
    encode2_2 = residual_block(filters_2[1])(encode2_2)
    down2_2 = down_block(filters_2[2])(encode2_2)

    encode2_3 = residual_block(filters_2[2])(down2_2)
    encode2_3 = residual_block(filters_2[2])(encode2_3)
    down2_3 = down_block(filters_2[3])(encode2_3)

    encode2_4 = residual_block(filters_2[3])(down2_3)
    encode2_4 = residual_block(filters_2[3])(encode2_4)
    encode2_4 = residual_block(filters_2[3])(encode2_4)
    encode2_4 = residual_block(filters_2[3])(encode2_4)
    

    """
    ===============================================================================================================
    U-Net 2 Decoder 1 (Deconvolution)
    """
    up2_3_1 = up_block_1(filters_2[2])(encode2_4)
    decode2_3_1 = Add()([up2_3_1, encode2_3])
    decode2_3_1 = residual_block(filters_2[2])(decode2_3_1)
    
    up2_2_1 = up_block_1(filters_2[1])(decode2_3_1)
    decode2_2_1 = Add()([up2_2_1, encode2_2])
    decode2_2_1 = residual_block(filters_2[1])(decode2_2_1)
    

    up2_1_1 = up_block_1(filters_2[0])(decode2_2_1)
    decode2_1_1 = Add()([up2_1_1, encode2_1])
    decode2_1_1 = residual_block(filters_2[0])(decode2_1_1)

    final2_1 = final_block(filters = num_class, stage = '2_1')(decode2_1_1)


    """
    ===============================================================================================================
    U-Net 2 Decoder 2 (Bilinear Interpolation)
    """
    up2_3_2 = up_block_2(filters_2[2])(encode2_4)
    decode2_3_2 = Add()([up2_3_2, encode2_3])
    decode2_3_2 = residual_block(filters_2[2])(decode2_3_2)
    

    up2_2_2 = up_block_1(filters_2[1])(decode2_3_2)
    decode2_2_2 = Add()([up2_2_2, encode2_2])
    decode2_2_2 = residual_block(filters_2[1])(decode2_2_2)
    

    up2_1_2 = up_block_1(filters_2[0])(decode2_2_2)
    decode2_1_2 = Add()([up2_1_2, encode2_1])
    decode2_1_2 = residual_block(filters_2[0])(decode2_1_2)
    
    final2_2 = final_block(filters = num_class, stage = '2_2')(decode2_1_2)
    

    """
    ===============================================================================================================
    Model Shape
    """
    # Format Setting
    space = "{:15}{:20}"

    # Encoder 1
    print(space.format('img_input_1:', str(img_input_1.shape)))
    print(space.format('conv1:', str(conv1.shape)))
    print(space.format('encode1_1:', str(encode1_1.shape)))
    print(space.format('down1_1:', str(down1_1.shape)))
    print(space.format('encode1_2:', str(encode1_2.shape)))
    print(space.format('down1_2:', str(down1_2.shape)))
    print(space.format('encode1_3:', str(encode1_3.shape)))
    print(space.format('down1_3:', str(down1_3.shape)))
    print(space.format('encode1_4:', str(encode1_4.shape)), '\n')

    # Decoder 1
    print(space.format('up1_3:', str(up1_3.shape)))
    print(space.format('decode1_3:', str(decode1_3.shape)))
    print(space.format('up1_2:', str(up1_2.shape)))
    print(space.format('decode1_2:', str(decode1_2.shape)))
    print(space.format('up1_1:', str(up1_1.shape)))
    print(space.format('decode1_1:', str(decode1_1.shape)))
    print(space.format('final1:', str(final1.shape)), '\n\n')

    # Encoder 2
    print(space.format('img_input_2:', str(img_input_2.shape)))
    print(space.format('conv2:', str(conv2.shape)))
    print(space.format('encode2_1:', str(encode2_1.shape)))
    print(space.format('down2_1:', str(down2_1.shape)))
    print(space.format('encode2_2:', str(encode2_2.shape)))
    print(space.format('down2_2:', str(down2_2.shape)))
    print(space.format('encode2_3:', str(encode2_3.shape)))
    print(space.format('down2_3:', str(down2_3.shape)))
    print(space.format('encode2_4:', str(encode2_4.shape)), '\n')

    # Decoder 2-1 (Deconvolutional)
    print(space.format('up2_3_1:', str(up2_3_1.shape)))
    print(space.format('decode2_3_1:', str(decode2_3_1.shape)))
    print(space.format('up2_2_1:', str(up2_2_1.shape)))
    print(space.format('decode2_2_1:', str(decode2_2_1.shape)))
    print(space.format('up2_1_1:', str(up2_1_1.shape)))
    print(space.format('decode2_1_1:', str(decode2_1_1.shape)))
    print(space.format('final2_1:', str(final2_1.shape)), '\n')

    # Decoder 2-2 (Bilinear Interpolation)
    print(space.format('up2_3_2:', str(up2_3_2.shape)))
    print(space.format('decode2_3_2:', str(decode2_3_2.shape)))
    print(space.format('up2_2_2:', str(up2_2_2.shape)))
    print(space.format('decode2_2_2:', str(decode2_2_2.shape)))
    print(space.format('up2_1_2:', str(up2_1_2.shape)))
    print(space.format('decode2_1_2:', str(decode2_1_2.shape)))
    print(space.format('final2_2:', str(final2_2.shape)), '\n')
    
    
    """
    ===============================================================================================================
    Model
    """
    model = Model(img_input_1, [final1, final2_1, final2_2])
        

    return model


"""
===================================================================================================================
Main Function
"""
if __name__ == '__main__':

    model = cascaded_unet(img_size = 128, color_type = 4, num_class = 3)
    print('Parameters:', model.count_params())