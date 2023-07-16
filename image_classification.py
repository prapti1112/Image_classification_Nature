from glob import glob
import logging
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model

class Image_Classification:
    def __init__(self, model_name="vgg19") -> None:
        self.IMAGE_SIZE = 224
        logging.info(f"Starting image classification with {model_name}")

        self.weights_path = r"C:\Users\prapt\Desktop\Deep Learning\Image_classification_Nature\pretrained_weights"
        if model_name == "vgg19":
            from classifiers.vgg import VGG
            self.model = VGG()
            self.weights_path = os.path.join(self.weights_path, "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")

    def load_data(self, data_path):
        """
            Function to load a dataset given the path to data
            @param[in]: data_path(String/PathLike)
            ---
            @param[out]: dataset[tf.data.dataset]
        """
        logging.info(f"Loading data from: {data_path}")
        dataset = tf.keras.utils.image_dataset_from_directory(data_path, image_size=(self.IMAGE_SIZE, self.IMAGE_SIZE))
        dataset = dataset.prefetch(tf.data.AUTOTUNE) # type: ignore

        return dataset

    def train(self, data_path, model_save_path):
        """
            Function to train an image classifier
            @param[in]: data_path(String/PathLike) - Path to where the data is store.
            In the current example it is assumed that the data is stored in the following manner:
            root_folder
            | --- train
            |       | --- class1
            |               | --- image_class1.png
            |               | --- image_class2.png
            |               | --- image_class3.png
            |                      :
            |       | --- class2
            |           :
            | --- val
            |       | --- class1
            |               | --- image_class1.png
            |                      :
            |       | --- class2
            |           :
        """
        # TODO: Validate the data path entered
        with tf.device('/device:gpu:0'):
            train_dataset = self.load_data(os.path.join(data_path, "train"))
            validation_dataset = self.load_data(os.path.join(data_path, "val"))

        model = self.model.build_model((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), 10)
        model.summary()

        model.compile(optimizer=self.model.compiler_params["optimizer"], loss=self.model.compiler_params["loss"], metrics=self.model.compiler_params["metrics"] )

        with tf.device('/device:gpu:0'):
            model.fit( train_dataset, validation_data=validation_dataset, epochs=20000, use_multiprocessing=True,
                      callbacks=[ModelCheckpoint( os.path.join(model_save_path, "vgg19_{epoch:03d}-{val_loss:.4f}.h5" )) ] )
        
        model.save( os.path.join(model_save_path, "vgg19_final.h5") )
        
    def infer(self, data_path, model_path):
        logging.info(f"Loading model from: {model_path}")
        model = load_model( model_path )

        logging.debug("Reading images and preprocessing them")
        images = np.array([ cv2.resize(cv2.imread(imagePath), ( self.IMAGE_SIZE, self.IMAGE_SIZE )) for imagePath in glob( os.path.join(data_path, "*.jpg") )])
        logging.debug(f"Input shape: {images.shape}")
        prediction = model.predict(images)
        logging.debug( f"Prediction: {prediction}\n" )
            


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    classifier = Image_Classification()
    classifier.train(r"C:\Users\prapt\Desktop\Deep Learning\Image_classification_Nature\data\inaturalist_12K", r"C:\Users\prapt\Desktop\Deep Learning\Image_classification_Nature\models")
    classifier.infer(r"C:\Users\prapt\Desktop\Deep Learning\Image_classification_Nature\data\inaturalist_12K_inference_data", r"C:\Users\prapt\Desktop\Deep Learning\Image_classification_Nature\models\vgg19_final.h5")