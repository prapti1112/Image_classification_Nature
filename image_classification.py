import logging

class Image_Classification:
    def __init__(self, model_name="vgg19") -> None:
        self.IMAGE_SIZE = 224
        logging.info(f"Starting image classification with {model_name}")

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
        logging.info(f"Loading data from {data_path}")

    def infer(self):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    classifier = Image_Classification()
    classifier.train(r"C:\Users\prapt\Desktop\Deep Learning\Image_classification_Nature\data\inaturalist_12K")