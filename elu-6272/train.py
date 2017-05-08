import pickle
import time
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

model.train_images(utils.normal_gray(X_train), y_train, utils.normal_gray(X_valid), y_valid,
                   checkpoint=config.checkpoint, trains=config.trainings, epochs=config.epochs,
                   dropout_keep=config.drr, learning_rate=config.lr, accept=config.accept,
                   batch_size=config.batch)
