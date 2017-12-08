import pickle
import utils
import model
from config import config

with open(config.test, mode='rb') as f:
    test_data = pickle.load(f)

samples, labels = test_data['features'], test_data['labels']

print("Number of testing examples =", len(samples))
print("Image data shape =", samples.shape[1:])

model.test_images(utils.normal_gray(samples), labels, config.checkpoint)
