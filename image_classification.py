from glob import glob
import logging
import math
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
import tqdm

import wandb
from wandb.keras import WandbCallback


class Image_Classification:
    def __init__(self, model_name="vgg19") -> None:
        self.IMAGE_SIZE = 112
        logging.info(f"Starting image classification with {model_name}")

        self.weights_path = r"pretrained_weights"
        if model_name == "vgg19":
            from classifiers.vgg import VGG
            self.model = VGG()
            self.weights_path = os.path.join(self.weights_path, "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
        if model_name == "resnet152":
            from classifiers.resnet import ResNet
            self.model = ResNet()

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

    def train_step(self, x, y, model, optimizer, loss_fn, train_acc_metric):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_acc_metric.update_state(y, logits)

        return loss_value

    def test_step(self, x, y, model, loss_fn, val_acc_metric):
        val_logits = model(x, training=False)
        loss_value = loss_fn(y, val_logits)
        val_acc_metric.update_state(y, val_logits)

        return loss_value

    def sweep_train(self, train_dataset, val_dataset, model, optimizer, loss_fn, train_acc_metric, val_acc_metric, epochs=10):
        for epoch in range(epochs):
            print( f"Epoch: {epoch}/{epoch}" )

            train_loss, val_loss = [], []

            for step, (x_batch_train, y_batch_train) in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
                loss_value = self.train_step(x_batch_train, y_batch_train, 
                                    model, optimizer, 
                                    loss_fn, train_acc_metric)
                train_loss.append(float(loss_value))
            
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                val_loss_value = self.test_step(x_batch_val, y_batch_val, 
                                        model, loss_fn, val_acc_metric)
                val_loss.append(float(val_loss_value))
            
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            val_acc = val_acc_metric.result()
            print("Validation acc: %.4f" % (float(val_acc),))

            train_acc_metric.reset_states()
            val_acc_metric.reset_states()

            wandb.log({'epochs': epoch,
                    'loss': np.mean(train_loss),
                    'acc': float(train_acc), 
                    'val_loss': np.mean(val_loss),
                    'val_acc':float(val_acc)})
    
    def train_for_sweep(self):

        with tf.device('/device:gpu:0'):
            train_dataset = self.load_data(os.path.join(self.data_path, "train"))
            validation_dataset = self.load_data(os.path.join(self.data_path, "val"))

        model = self.model.build_model((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), 10)
        model.summary()

        model.compile(optimizer=self.model.compiler_params["optimizer"], loss=self.model.compiler_params["loss"], metrics=self.model.compiler_params["metrics"] )

        wandb.init( config={ 'batch_size': 64, 'learning_rate': 0.01 }, project="nature_classification")

        optimizer = SGD(learning_rate=wandb.config.learning_rate)
        loss_fn = SparseCategoricalCrossentropy(from_logits=True)

        # Prepare the metrics.
        train_acc_metric = SparseCategoricalAccuracy()
        val_acc_metric = SparseCategoricalAccuracy()

        with tf.device('/device:gpu:0'):
            self.sweep_train(train_dataset,
            validation_dataset, 
            model,
            optimizer,
            loss_fn,
            train_acc_metric,
            val_acc_metric,
            epochs=wandb.config.epochs)

    def sweep_caller(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

        sweep_config = {
            "method": "random",
            "metric": {
                "name": "val_loss",
                "goal": "minimize"
            },
            'parameters': {
                'optimizer': {
                    'values': [ 'adam', 'sgd' ],
                    },
                'fc_layer_size': {
                    'values': [128, 256, 512]
                },
                'dropout': {
                    'values': [0.3, 0.4, 0.5]
                },
                'epochs': {
                    'values': [10, 20, 50, 80, 120],
                },
                'learning_rate':{
                    'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]
                },
                'batch_size': {
                    'values': [8, 16, 32, 64, 128]
                }
            }
        }
        
        sweep_id = wandb.sweep(sweep_config, project="Nature-Image-Classification")
        print( f"Sweep id: {sweep_id}" )
        wandb.agent(sweep_id, function=self.train_for_sweep, count=10)

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

        model = self.model.build_model((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), 10, 0.4)
        model.summary()

        model.compile(optimizer=self.model.compiler_params["optimizer"], loss=self.model.compiler_params["loss"], metrics=self.model.compiler_params["metrics"] )

        with tf.device('/device:gpu:0'):
            model.fit( train_dataset, validation_data=validation_dataset, epochs=80, use_multiprocessing=True, batch_size=16,
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
    classifier = Image_Classification(model_name="vgg19")
    # classifier.sweep_caller(r"data\inaturalist_12K", r"models")
    classifier.train( r"data\inaturalist_12K", r"models" )
    # classifier.infer(r"data\inaturalist_12K_inference_data", r"models\vgg19_final.h5")