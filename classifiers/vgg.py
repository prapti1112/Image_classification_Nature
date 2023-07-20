"""
    This is a code to recreate the model architecture given in the 
    Paper: VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION [https://arxiv.org/pdf/1409.1556v6.pdf]
"""

import logging
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Layer, Dropout
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.optimizers import gradient_descent_v2

class VGG:
    def __init__(self) -> None:
        self.compiler_params = {
            "optimizer": gradient_descent_v2.SGD(learning_rate=0.01,  momentum=0.0, decay=0.1),
            "loss": SparseCategoricalCrossentropy(from_logits=True),
            "metrics": ['accuracy'],
            "learning_rate": 0.1,
        }
        logging.info("Initializing vgg network")
    
    class Conv_Block(Layer):
        def __init__(self, num_filters, block_num, conv_per_block=2):
            super().__init__()
            self.num_filters = num_filters
            self.block_number = block_num
            self.conv_per_block = conv_per_block
        
        def __call__(self, input_tensor):
            x = Conv2D(self.num_filters, 3, activation="relu", name = f"Conv_{self.block_number}_1")(input_tensor)
            for i in range(2, self.conv_per_block+1):
                x = Conv2D( self.num_filters, 3, activation="relu", padding="same", name = f"Conv_{self.block_number}_{i}" )(x)
            x = MaxPool2D(strides=2, padding="same", name=f"MaxPooling_{self.block_number}")(x)
            
            return x

    def build_model(self, input_size, num_classes, drop_out_probability=0.5):
        """
            This function build the vgg-19 model using the function api
            @param[in]: input_size(int)
            @param[in]: num_classes(int) - output classes
            ---
            @param[out]: model(tf.keras.Model)
        """
        logging.debug("Starting to build vgg19 model")

        input = Input(shape = input_size)
        
        x = self.Conv_Block(64, 1)(input) 
        x = self.Conv_Block(128, 2)(x) 
        x = self.Conv_Block(256, 3, 3)(x)
        x = self.Conv_Block(512, 4, 3)(x)
        x = self.Conv_Block(512, 5, 3)(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(drop_out_probability)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(drop_out_probability)(x)
        x = Dense(num_classes, activation='softmax')(x)
        model = Model( inputs=input, outputs=x )
        
        return model