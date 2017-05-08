'''
This package contains functions for creating, saving, loading, training, testing, and
evaluating convolutional neural network
'''
import glob
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

def leakyReLu(inputs, alpha=0.05, name=None):
    '''
    Leaky ReLu from Keras's Tensorflow backend implementation

    Arguments:
    alpha: slope of negative section.
    '''
    return tf.subtract(tf.nn.relu(inputs), alpha*tf.nn.relu(-inputs), name=name)

def kernel_stride(width, data_format='NCHW'):
    '''
    Create kernel or strides fot the given data format

    Arguments:
    s: the kernel or stride size
    data_format: the data format, either 'NCHW' or 'NHWC'. Default is 'NCHW'
    '''
    if data_format == 'NCHW':
        return [1, 1, width, width]
    else:
        return [1, width, width, 1]

def weights(shape, name, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    '''
    Create a new weight variable

    Arguments:
    shape: shape of the variable
    name: name of the variable
    initializer: initializer for the variable, default is xavier initializer
    '''
    return tf.get_variable(name, shape=shape, initializer=initializer)

def biases(shape, name, initializer=tf.zeros_initializer()):
    '''
    Create a new bias variable

    Arguments:
    shape: shape of the variable
    name: name of the variable
    initializer: initializer for the variable, default is zeros initializer
    '''
    return tf.get_variable(name, shape=shape, initializer=initializer)

def layer_counter():
    '''
    Return the current layer count, this is used to generate unique names
    '''
    if "index" not in layer_counter.__dict__:
        layer_counter.index = 0
    layer_counter.index += 1
    return layer_counter.index

def conv2d(inputs, shape, strides, data_format='NCHW', padding='SAME',
           activation=None):
    '''
    Create a new convolution layer

    Arguments:
    inputs: the input
    shape: shape of the kernel
    strides: the strides
    data_format: the data format, either 'NCHW' or 'NHWC'. Default is 'NCHW'
    padding: the padding
    '''
    layer = layer_counter()
    weight = weights(name='Wconv{}'.format(layer), shape=shape)
    bias = biases(name='Bconv{}'.format(layer), shape=shape[-1:])
    logits = tf.nn.conv2d(inputs, weight, strides=strides, data_format=data_format, padding=padding)
    logits = tf.nn.bias_add(logits, bias, data_format=data_format)
    if activation is None:
        logits = leakyReLu(logits, alpha=0.02, name='conv{}'.format(layer))
    else:
        logits = activation(logits, alpha=0.02, name='conv{}'.format(layer))
    return logits

def fcnn(inputs, shape, keep_prob=None):
    '''
    Create a fully connection network layer

    Arguments:
    inputs: the inputs layer
    shape: shape of the kernel
    keep_prob: dropout's keep proability. The layer if not have dropout if keep_prob is None
    '''
    layer = layer_counter()
    weight = weights(name='Wfcnn{}'.format(layer), shape=shape)
    bias = biases(name='Bfcnn{}'.format(layer), shape=shape[-1:])
    logits = tf.add(tf.matmul(inputs, weight), bias)
    logits = tf.nn.elu(logits, name='fcn{}'.format(layer))
    if keep_prob != None:
        logits = tf.nn.dropout(logits, keep_prob)
    return logits

def cnn(features, classes, keep_prob=None, data_format='NCHW'):
    '''
    Create our 7-layer CNN model composed of 4 convolutiuon layers and 3 fully connected layers

    Arguments:
    x: the inputs layer
    classes: the number of classes
    keep_prob: dropout's keep proability. The layer if not have dropout if keep_prob is None
    data_format: the data format, either 'NCHW' or 'NHWC'. Default is 'NCHW'
    '''
    kernel_stride2 = kernel_stride(2, data_format=data_format)

    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x16.
    logits = conv2d(features, shape=(3, 3, 1, 16), strides=[1, 1, 1, 1], data_format=data_format)

    # Layer 2: Convolutional. Input = 32x32x16 output = 32x32x32
    logits = conv2d(logits, shape=(3, 3, 16, 32), strides=[1, 1, 1, 1], data_format=data_format)

    # Layer 3: Convolutional. Input = 32x32x32 output = 32x32x64
    logits = conv2d(logits, shape=(3, 3, 32, 64), strides=[1, 1, 1, 1], data_format=data_format)

    # Pooling. Input = 32x32x64. Output = 16x16x64.
    logits = tf.nn.max_pool(logits, ksize=kernel_stride2, strides=kernel_stride2,
                            data_format=data_format, padding='SAME')

    # Layer 4: Convolutional. Input = 16x16x64, Output = 14x14x128.
    logits = conv2d(logits, shape=(3, 3, 64, 128), strides=[1, 1, 1, 1], data_format=data_format,
                    padding='VALID')

    # Pooling. Input = 14x14x128. Output = 7x7x128.
    logits = tf.nn.max_pool(logits, ksize=kernel_stride2, strides=kernel_stride2,
                            data_format=data_format, padding='SAME')

    # Flatten. Input = 7x7x128. Output = 6272.
    logits = flatten(logits)

    logits = tf.nn.dropout(logits, keep_prob)

    # Layer 5: Fully Connected. Input = 6272. Output = 1600.
    logits = fcnn(logits, shape=(logits.get_shape()[-1], 1600), keep_prob=keep_prob)

    # Layer 6: Fully Connected. Input = 1600. Output = 400.
    logits = fcnn(logits, shape=(1600, 400), keep_prob=keep_prob)

    # Layer 7: Fully Connected. Input = 400. Output = classes, which is 42.
    logits = fcnn(logits, shape=(400, classes))

    return logits

def vgg16(features, classes, keep_prob=None, data_format='NCHW'):
    '''
    Create vgg16

    Arguments:
    x: the inputs layer
    classes: the number of classes
    keep_prob: dropout's keep proability. The layer if not have dropout if keep_prob is None
    data_format: the data format, either 'NCHW' or 'NHWC'. Default is 'NCHW'
    '''
    s1x1 = [1, 1, 1, 1]
    s2x2 = kernel_stride(2, data_format=data_format)

    # Layer 1: Convolutional. Input = 224x224x3. Output = 224x224x64.
    logits = conv2d(features, shape=(3, 3, 3, 64), strides=s1x1, data_format=data_format)
    # Layer 2: Convolutional. Input = 224x224x64 output = 224x224x64
    logits = conv2d(logits, shape=(3, 3, 64, 64), strides=s1x1, data_format=data_format)
    logits = tf.nn.max_pool(logits, ksize=s2x2, strides=s2x2,
                            data_format=data_format, padding='SAME')

    # Layer 3: Convolutional. Input = 112x112x64. Output = 112x112x128.
    logits = conv2d(logits, shape=(3, 3, 64, 128), strides=s1x1, data_format=data_format)
    # Layer 4: Convolutional. Input = 112x112x128 output = 112x112x128
    logits = conv2d(logits, shape=(3, 3, 128, 128), strides=s1x1, data_format=data_format)
    logits = tf.nn.max_pool(logits, ksize=s2x2, strides=s2x2,
                            data_format=data_format, padding='SAME')

    # Layer 5: Convolutional. Input = 56x56x128. Output = 56x56x256.
    logits = conv2d(logits, shape=(3, 3, 128, 256), strides=s1x1, data_format=data_format)
    # Layer 6: Convolutional. Input = 56x56x256 output = 56x56x256
    logits = conv2d(logits, shape=(3, 3, 256, 256), strides=s1x1, data_format=data_format)
    # Layer 7: Convolutional. Input = 56x56x256 output = 56x56x256
    logits = conv2d(logits, shape=(3, 3, 256, 256), strides=s1x1, data_format=data_format)
    logits = tf.nn.max_pool(logits, ksize=s2x2, strides=s2x2,
                            data_format=data_format, padding='SAME')

    # Layer 8: Convolutional. Input = 28x28x256. Output = 28x28x512.
    logits = conv2d(logits, shape=(3, 3, 256, 512), strides=s1x1, data_format=data_format)
    # Layer 9: Convolutional. Input = 28x28x512 output = 28x28x512
    logits = conv2d(logits, shape=(3, 3, 512, 512), strides=s1x1, data_format=data_format)
    # Layer 10: Convolutional. Input = 28x28x512 output = 28x28x512
    logits = conv2d(logits, shape=(3, 3, 512, 512), strides=s1x1, data_format=data_format)
    logits = tf.nn.max_pool(logits, ksize=s2x2, strides=s2x2,
                            data_format=data_format, padding='SAME')

    # Layer 11: Convolutional. Input = 14x14x512. Output = 14x14x512.
    logits = conv2d(logits, shape=(3, 3, 512, 512), strides=s1x1, data_format=data_format)
    # Layer 12: Convolutional. Input = 28x28x512 output = 28x28x512
    logits = conv2d(logits, shape=(3, 3, 512, 512), strides=s1x1, data_format=data_format)
    # Layer 13: Convolutional. Input = 28x28x512 output = 28x28x512
    logits = conv2d(logits, shape=(3, 3, 512, 512), strides=s1x1, data_format=data_format)

    logits = tf.nn.max_pool(logits, ksize=s2x2, strides=s2x2,
                            data_format=data_format, padding='SAME')

    # Flatten. Input = 7x7x512. Output = 25088.
    logits = flatten(logits)

    logits = tf.nn.dropout(logits, keep_prob)

    # Layer 14: Fully Connected. Input = 14x14x512. Output = 4096.
    logits = fcnn(logits, shape=(logits.get_shape()[-1], 4096), keep_prob=keep_prob)

    # Layer 15: Fully Connected. Input = 4096. Output = 4096.
    logits = fcnn(logits, shape=(4096, 4096), keep_prob=keep_prob)

    # Layer 16: Fully Connected. Input = 4096. Output = classes, which is 42.
    logits = fcnn(logits, shape=(4096, classes))

    return logits


class CNNModel:
    '''
    Encapsulate c CNN model
    '''
    def __init__(self, features, labels, classes, keep_prob, logits):
        '''
        features: the features placeholder
        labels: the labels placeholder
        classes: number of classes
        keep_prob: the placeholder for dropout keep proability
        logits: the tensorflow logitistic model
        '''
        self.features = features
        self.labels = labels
        self.classes = classes
        self.keep_prob = keep_prob
        self.logits = logits

def create_model(feature_shape, classes, feature_dtype=tf.float32, label_dtype=tf.int32,
                 factory=None):
    '''
    Re-initializeTensorflow environment and create the core network model

    Arguments:
    feature_shape: shape of the features
    classes" classes of the features
    feature_dtype: features's data type, default is float32
    label_dtype: label's data type, default is int32
    '''
    tf.reset_default_graph()
    if feature_shape[0] < feature_shape[1] and feature_shape[0] < feature_shape[2]:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    features = tf.placeholder(feature_dtype, (None,) + feature_shape)
    labels = tf.placeholder(label_dtype, None)
    keep_prob = tf.placeholder(tf.float32)
    if factory is None:
        logits = cnn(features, classes, keep_prob=keep_prob, data_format=data_format)
    else:
        logits = factory(features, classes, keep_prob=keep_prob, data_format=data_format)
    tf.add_to_collection("features", features)
    tf.add_to_collection("labels", labels)
    tf.add_to_collection("keep_prob", keep_prob)
    tf.add_to_collection("model", logits)
    return CNNModel(features, labels, classes, keep_prob, logits)

def save_current_model(name):
    '''
    Save the current Tensorflow model

    Arguments:
    name: checkpoint name of the model
    '''
    tf.train.export_meta_graph(filename=name+'.meta', clear_devices=True)

def load_model(name, classes):
    '''
    Loade current Tensorflow model

    Arguments:
    name: checkpoint name of the model
    '''
    tf.reset_default_graph()
    tf.train.import_meta_graph(name + '.meta')
    graph = tf.get_default_graph()
    features = graph.get_collection('features')[0]
    labels = graph.get_collection('labels')[0]
    keep_prob = graph.get_collection('keep_prob')[0]
    logits = graph.get_collection('model')[0]
    return CNNModel(features, labels, classes, keep_prob, logits)

def save_trained_model(session, checkpoint, step=None, saver=None):
    '''
    Save the trained model

    Arguments:
    session: the current tensorflow training session
    checkpoint: the checkpoint (the file name prefix where the trained model is saved)
    step: the step (epoch) in which the model was trained
    saver: the saver object to use, a new saver will be used if None is given
    '''
    if saver is None:
        saver = tf.train.Saver(tf.trainable_variables())
    saver.save(session, checkpoint, global_step=step)

def load_trained_model(session, checkpoint):
    '''
    Loade trained model

    Arguments:
    session: the tensorflow session to restore the trained model to
    checkpoint: the checkpoint (the file name prefix where the trained model is to be loaded)
    classes: classes of the model's labels
    '''
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    saver.restore(session, checkpoint)
    graph = tf.get_default_graph()
    features = graph.get_collection('features')[0]
    labels = graph.get_collection('labels')[0]
    keep_prob = graph.get_collection('keep_prob')[0]
    logits = graph.get_collection('model')[0]
    assert logits is not None, "Invalid model, missing logits!"
    assert features is not None, "Invalid model, missing features!"
    assert labels is not None, "Invalid model, missing labels!"
    assert keep_prob is not None, "Invalid model, missing dropout retention!"
    return CNNModel(features, labels, logits.get_shape()[-1], keep_prob, logits)

def find_trained_model(checkpoint):
    '''
    Find the list of checkpoint models, return the checkpoint names of the trained model of format:
    'path/checkpoint_prefix-nn'

    Arguments:
    checkpoint: checkpoint's path and base name
    '''
    files = glob.glob('{0}-*.meta'.format(checkpoint))
    if files is None or len(files) == 0:
        print("No trained model found with {0}!".format(checkpoint))
        return None
    # Else, extract the checkpoint to use
    files.sort(key=os.path.getmtime)
    return [file[:-5] for file in reversed(files)]

def find_tensor(model, name):
    '''
    Find tensor of the given name from current graph. It can then be used to eval inputs,
    or compose a network from it.

    Arguments:
    Model: either an instance of CNNMOdel or a tensorflow operation
    name: name of the tensor

    Return: the tensor if found, or (first output, outputs) of an operation if found,
            or None otherwise
    '''
    if isinstance(model, CNNModel):
        model = model.logits

    graph = model.graph

    # First find the operation
    try:
        tensor = graph.get_operation_by_name(name)
    except KeyError: # Unable to find an operation, try to find a tensor
        try:
            tensor = graph.get_tensor_by_name(name)
        except KeyError: # nothing was found
            pass
        else: # Found a tensor
            return tensor
    else:
        if tensor is not None: # we have found an operation
            if len(tensor.outputs) > 1:
                return tensor.outputs[0], tensor.outputs
            else:
                return tensor.outputs[0]

def create_training(model, learning_rate, global_step=None):
    '''
    Create the training model from the core network model. It uses adam optimizer for training

    Arguments:
    model: the core network model
    learning_rate: the learning rate
    global_step: the global step, default is None
    '''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=model.labels, logits=model.logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-7)
    return optimizer.minimize(loss_operation, global_step=global_step)

def create_evaluation(model):
    '''
    Create evaluation model from the core model

    Argument:
    model: the CNN model
    '''
    correct_prediction = tf.equal(tf.arg_max(model.logits, 1), model.labels)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def create_prediction(model):
    '''
    Create prediction model from the core model
    Parameters:
    model: the CNN model
    '''
    correct_prediction = tf.argmax(model.logits, 1)
    return correct_prediction

def create_softmax_evaluation(model, top=5):
    '''
    Create softmax evaluation model from the core model

    Arguments:
    model: the CNN model
    top: the number of top results to evaluate
    '''
    softmax = tf.nn.softmax(model.logits)
    return softmax, tf.nn.top_k(softmax, top)

def evaluate(session, features, labels, operation, model, batch_size=256, preprocessor=None):
    '''
    Evaluate the current training model

    Arguments:
    session: the current session
    features: the features to evaluate
    labels: labels of the features
    operation: the evaluation operation
    model: the CNN model
    batch_size: the batch processing size, default is 256
    '''
    num_examples = len(features)
    total_accuracy = 0
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = features[offset:offset+batch_size], labels[offset:offset+batch_size]
        if preprocessor is not None:
            batch_x, batch_y = preprocessor(batch_x, batch_y)
        accuracy = session.run(operation, feed_dict={model.features: batch_x, model.labels: batch_y,
                                                     model.keep_prob:1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def train_model(features, labels, v_samples, v_labels, checkpoint, trains=1, epochs=60, dropout_keep_prob=0.85,
                learning_rate=0.0006, batch_size=256, accept=0.98, factory=None, preprocessor=None):
    '''
    This is a generator for generating the best training models. Create and train a CNN model with
    the features, and saves the best training models in checkpoint for prediction operations. The
    generator will yield everytime a best training model is generated.

    Arguments:
    features: the features to train
    labels: labels of the features
    v_samples: the validation data
    v_labels: the labels of the validation data
    trains: number of trainings to perform for the best model
    epochs: the number of epochs for each training
    dropout_keep_prob: the keep probability of drop out
    learning_rate: thje learning rate
    batch_size: the batch size
    accept: the minimal accuracy to accept for a training model
    checkpoint: the base name of the checkpoints where the best training models will be saved
    '''
    save_id = 0
    count = 0
    classes = int(max(labels) - min(labels)) + 1
    n_samples = len(features)

    model = create_model(features.shape[1:] if preprocessor is None else preprocessor.shape,
                         classes, label_dtype=tf.int64, factory=factory)

    training_operation = create_training(model, learning_rate)
    accuracy_operation = create_evaluation(model)
    while count < trains:
        count += 1
        with tf.Session() as session:
            print("Training {} ...".format(count))
            session.run(tf.global_variables_initializer())
            for i in range(epochs):
                features, labels = shuffle(features, labels)
                for offset in range(0, n_samples, batch_size):
                    end = offset + batch_size
                    batch_x, batch_y = features[offset:end], labels[offset:end]
                    if preprocessor is not None:
                        batch_x, batch_y = preprocessor(batch_x, batch_y)
                    session.run(training_operation, feed_dict={model.features: batch_x,
                                                               model.labels: batch_y,
                                                               model.keep_prob: dropout_keep_prob})

                accuracy = evaluate(session, v_samples, v_labels, accuracy_operation, model,
                                    batch_size=batch_size, preprocessor=preprocessor)
                print("EPOCH {0:3}: Accuracy = {1:.4f}".format(i+1, accuracy))
                if accuracy >= accept:
                    save_id += 1
                    for _ in range(3):
                        try:
                            save_trained_model(session, checkpoint, step=save_id)
                            break
                        except:
                            print("Failed to save trained model!")
                            time.sleep(2)
                    yield accuracy, save_id

def train_images(samples, labels, v_samples, v_labels, trains=2, epochs=200, dropout_keep=0.5,
                 learning_rate=0.001, batch_size=250, accept=0.994, checkpoint="./trained/model",
                 factory=None, preprocessor=None):
    '''
	Train model using train_model generator

    Arguments:
	samples: the training samples
	labels: the training labels
	v_samples: the validation samples
	v_labels: the validation labels
	trains: the number of trainings to perform
    epochs: the number of epochs for each training
    dropout_keep: the keep probability of drop out
    learning_rate: thje learning rate
    batch_size: the batch size
    accept: the minimal accuracy to accept for a training model
    checkpoint: path and base name where the checkpoints will be stored
	'''
    print("Number of training examples =", len(samples))
    print("Number of validation examples =", len(v_samples))
    print("Image data shape =", samples.shape[1:])
    print("Number of classes =", max(labels) - min(labels) + 1)

    # Reshape the training and validation data for NCHW format
    if preprocessor is None:
        print("No preprocessor is given, preprocess all images ...")
        samples = samples.reshape((samples.shape[0], 1, samples.shape[1], samples.shape[2]))
        v_samples = v_samples.reshape((v_samples.shape[0], 1, v_samples.shape[1],
                                       v_samples.shape[2]))

    start_time = time.time()
    # Train the model with the training and validation data
    for accuracy, best_index in train_model(samples, labels, v_samples, v_labels,
                                            dropout_keep_prob=dropout_keep,
                                            trains=trains, checkpoint=checkpoint,
                                            learning_rate=learning_rate, epochs=epochs,
                                            batch_size=batch_size, accept=accept, factory=factory,
                                            preprocessor=preprocessor):
        print("Save model: {0}-{1}, accuracy: {2:.4f}".format(checkpoint, best_index, accuracy))

    print("Total training time: {:.3} seconds".format((time.time() - start_time)))

def test_images(images, labels, checkpoint, preprocessor=None):
    '''
    Test the images against the trained model in checkpoint

    Arguments:
    images: the images
    labels: the labels
    checkpoint: path and base name where the checkpoints will be stored
    '''
    print("Images: ", images.shape[0])
    print("Labels: ", labels)

    if preprocessor is None:
        images = images.reshape((images.shape[0], 1, images.shape[1], images.shape[2]))

    for trained in find_trained_model(checkpoint):
        tf.reset_default_graph()
        with tf.Session() as session:
            # Load the pre-trained model
            print("Loading trained model: {}".format(trained))
            model = load_trained_model(session, trained)

            print("Classes: ", model.classes)
            # Perdiction
            accuracy_operation = create_evaluation(model)
            # Create the accuracy perdiction
            accuracy = evaluate(session, images, labels, accuracy_operation, model, preprocessor=preprocessor)
            print("\tTest {1}, accuracy = {0:.4f}%".format(accuracy*100, trained))

def predict_images(images, labels, checkpoint, preprocessor=None):
    '''
    Evaluate the images against the trained model in checkpoint

    Arguments:
    images: the images
    labels: the labels
    checkpoint: path and base name where the checkpoints will be stored
    '''
    print("Evaluate images: ", images.shape[0])
    print("         Labels: ", labels)

    if preprocessor is None:
        print("No preprocessor is given, preprocess all images ...")
        images = images.reshape((images.shape[0], 1, images.shape[1], images.shape[2]))

    for trained in find_trained_model(checkpoint):
        tf.reset_default_graph()
        with tf.Session() as session:
            # Load the pre-trained model
            model = load_trained_model(session, trained)

            print("Classes: ", model.classes)
            prediction_operation = create_prediction(model)
            softmax_operation = create_softmax_evaluation(model, 3)
            # Evaluate the test data against the pre-trained data
            prediction = session.run(prediction_operation, feed_dict={model.features: images,
                                                                      model.labels: labels,
                                                                      model.keep_prob:1.0})
            # Evaluate the softmax performance
            _, softmax_performance = session.run(softmax_operation,
                                                 feed_dict={model.features: images,
                                                            model.labels: labels,
                                                            model.keep_prob:1.0})
            accuracy = float(np.sum(prediction == labels))/len(prediction)
            print("Prediction: {0}, accuracy: {1:.3f}%".format(prediction, accuracy * 100))
            print("Performance:\n", softmax_performance.values)
