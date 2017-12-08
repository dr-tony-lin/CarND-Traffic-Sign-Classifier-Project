import pickle
import numpy as np
import cv2
import utils
import model
from config import config

with open(config.train, mode='rb') as f:
    train_data = pickle.load(f)
with open(config.validate, mode='rb') as f:
    validation_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_valid, y_valid = validation_data['features'], validation_data['labels']

# How many unique classes/labels there are in the dataset.
classes = max(y_train) - min(y_train) + 1

def _preprocess(images, labels):
    images = np.array([cv2.resize(image, (224, 224)) for image in images])
    images = utils.normalize(images)
    images = images.reshape(images.shape[0], images.shape[3], 224, 224)
    return images, labels

_preprocess.shape=(3, 224,224)
model.train_images(X_train, y_train, X_valid, y_valid, checkpoint=config.checkpoint, 
                   trains=config.trainings, epochs=config.epochs, dropout_keep=config.drr,
                   learning_rate=config.lr, accept=config.accept, batch_size=config.batch,
                   factory=model.vgg16, preprocessor=_preprocess)
