import logging
import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

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
            model.fit( train_dataset.take(5), validation_data=validation_dataset.take(5), epochs=3, use_multiprocessing=True,
                      callbacks=[ModelCheckpoint( os.path.join(model_save_path, "vgg19_{epoch:03d}-{val_loss:.4f}.h5" ),  save_weights_only = True) ] )
        
    def infer(self):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    classifier = Image_Classification()
    classifier.train(r"C:\Users\prapt\Desktop\Deep Learning\Image_classification_Nature\data\inaturalist_12K", r"C:\Users\prapt\Desktop\Deep Learning\Image_classification_Nature\models")