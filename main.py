#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

REGULARIZE_CONST = 0.01  # Choose an appropriate one.

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    # load the model and weights of VGG16 which is used as encoder
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = sess.graph

    # load tensors from vgg model
    vgg_input_tensor_name = graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob_tensor_name = graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out_tensor_name = graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out_tensor_name = graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out_tensor_name = graph.get_tensor_by_name('layer7_out:0')
    
    return vgg_input_tensor_name, vgg_keep_prob_tensor_name, vgg_layer3_out_tensor_name, vgg_layer4_out_tensor_name, vgg_layer7_out_tensor_name
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # use the 1x1 convolutionals, so we don't mind the dimensions
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=REGULARIZE_CONST))
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=REGULARIZE_CONST))
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=REGULARIZE_CONST))


    # implement the decoder by upsampling the input to the original image size

    output = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, 2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=REGULARIZE_CONST))
    # retain the information of VGG encoder pooling layer 4
    output = tf.add(output, layer4_1x1)

    output = tf.layers.conv2d_transpose(output, num_classes, 4, strides=2, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=REGULARIZE_CONST))
    # retain the information of VGG encoder pooling layer 3
    output = tf.add(output, layer3_1x1)

    output = tf.layers.conv2d_transpose(output, num_classes, 16, strides=8, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=REGULARIZE_CONST))

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # reduce dimentions to have only the labels and the pixels
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # compute softmax cross entropy between logits and labels, picking the label with the highest value
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)


    #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    #accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #The losses are collected in the graph, and need to be added manually to the cost function
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy_loss = tf.reduce_mean(cross_entropy) + REGULARIZE_CONST * sum(reg_losses)


    # use Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    training_operation = optimizer.minimize(cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, saver):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    saver.save(sess, os.path.join('./data', 'FCN8s'), write_meta_graph=True)
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
             _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0001})

        print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss))
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            saver.save(sess, os.path.join('./data', 'FCN8s'), global_step=epoch, write_meta_graph=False)

    pass
#tests.test_train_nn(train_nn)  Added param "saver"


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function

        epochs = 100
        batch_size = 5

        # TF place holders:
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')


        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, training_operation, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=10)
        train_nn(sess, epochs, batch_size, get_batches_fn, training_operation, cross_entropy_loss, vgg_input,
                 correct_label, vgg_keep_prob, learning_rate, saver)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        # OPTIONAL: Apply the trained model to a video



if __name__ == '__main__':
    run()


def save_samples():

    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    with tf.Session() as sess:

        vgg_path = os.path.join(data_dir, 'vgg')

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')


        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, training_operation, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        init = tf.global_variables_initializer()
        sess.run(init)

        new_saver = tf.train.import_meta_graph('./data/FCN8s.meta')

        new_saver.restore(sess, './data/FCN8s-9')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        new_saver.restore(sess, './data/FCN8s-19')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        new_saver.restore(sess, './data/FCN8s-29')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        new_saver.restore(sess, './data/FCN8s-39')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        new_saver.restore(sess, './data/FCN8s-49')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        new_saver.restore(sess, './data/FCN8s-59')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        new_saver.restore(sess, './data/FCN8s-69')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        new_saver.restore(sess, './data/FCN8s-79')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        new_saver.restore(sess, './data/FCN8s-89')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)

        new_saver.restore(sess, './data/FCN8s-99')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, vgg_input)


save_samples()






