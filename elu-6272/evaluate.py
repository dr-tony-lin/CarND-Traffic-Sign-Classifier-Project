import glob
import re
import numpy as np
import cv2
import utils
import model

from config import config

files = glob.glob('../examples/traff-*.jpg')
images = np.array([cv2.cvtColor(cv2.imread(file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for file in files])
labels = [int(re.findall(r'\d+', file)[0]) for file in files]

print("Samples: ", images.shape)
print("Labels: ", labels)

model.predict_images(utils.normal_gray(images), labels, config.checkpoint)
