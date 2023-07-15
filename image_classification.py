import logging
import os
import tensorflow as tf

class Image_Classification:
    def __init__(self, model_name="vgg19") -> None:
        self.IMAGE_SIZE = 224
        logging.info(f"Starting image classification with {model_name}")

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

    def train(self, data_path):
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
        train_dataset = self.load_data(os.path.join(data_path, "train"))
        validation_dataset = self.load_data(os.path.join(data_path, "val"))

        # Testing datasets
        for (image, label) in train_dataset.take(3):
            print( image.shape, label )


    def infer(self):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    classifier = Image_Classification()
    classifier.train(r"C:\Users\prapt\Desktop\Deep Learning\Image_classification_Nature\data\inaturalist_12K")