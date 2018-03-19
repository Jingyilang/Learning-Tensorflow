import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from lenet import Lenet
# Parameters

#  Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0005, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-5, "learning rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch size")
tf.flags.DEFINE_integer("epoch", 50000, "epochs")
tf.flags.DEFINE_string("parameter_file", "checkpoint/variable.ckpt", "")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Prepare data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train, y_train = mnist.train.images, mnist.train.labels
x_val, y_val = mnist.validation.images, mnist.validation.labels
x_test, y_test = mnist.test.images, mnist.test.labels

# Training

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        lenet = Lenet(dropout_keep_prob=FLAGS.dropout_keep_prob,
                      learning_rate=FLAGS.learning_rate,
                      l2_reg_lambda=FLAGS.l2_reg_lambda)

        saver = tf.train.Saver(tf.global_variables())

        if os.path.exists(FLAGS.parameter_file):
            saver.restore(sess, FLAGS.paramter_file)
        else:
            sess.run(tf.global_variables_initializer())

        for i in range(FLAGS.epoch):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            if i % 500 == 0:
                dev_accuracy = sess.run(lenet.train_accuracy, feed_dict={
                    lenet.raw_input_image: x_val, lenet.raw_input_label: y_val
                })
                print("step %d, dev accuracy %g" % (i, dev_accuracy))
            sess.run(lenet.train_op, feed_dict={lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]})
            save_path = saver.save(sess, FLAGS.parameter_file)
