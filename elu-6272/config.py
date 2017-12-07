'''
Initialize command argument parsing and provide config for command line options
'''
import os
import tensorflow as tf

tf.app.flags.DEFINE_string('checkpoint', os.path.expanduser('~') + '/trained/model', "The path and base name of checkpoints")
tf.app.flags.DEFINE_string('data', 'data', "Sample data folder")
tf.app.flags.DEFINE_string('train', 'train.p', "Training samples file in pickle format (.p)")
tf.app.flags.DEFINE_string('validate', 'valid.p', "Validation samples file in pickle format  (.p)")
tf.app.flags.DEFINE_string('test', 'test.p', "Test samples file in pickle format  (.p)")

tf.app.flags.DEFINE_integer('trainings', 1, "The number of trainings.")
tf.app.flags.DEFINE_integer('epochs', 200, "The number of epochs per train.")
tf.app.flags.DEFINE_integer('batch', 256, "The batch size.")

tf.app.flags.DEFINE_float('lr', None, "The learning rate.")
tf.app.flags.DEFINE_float('drr', 0.5, "The dropout retention ratio.")
tf.app.flags.DEFINE_float('accept', 0.9, "The accepted training validation accuracy.")

config = tf.app.flags.FLAGS
config.train = config.data + "/" + config.train
config.validate = config.data + "/" + config.validate
config.test = config.data + "/" + config.test

def lrFromBatch():
    # The optimal learning rate for batch size 256 is 0.001, and the optimal rate for batch size 16 is about 0.0001.
    # Here we uses linear interpolation to obtain the learning rate for a given batch size
    config.lr = 0.0001 + (0.001 - 0.0001) * (config.batch - 16.0)/(256.0 - 16.0)
    return config.lr

if config.lr is None:
    lrFromBatch()
