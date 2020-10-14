# -*- coding: utf-8 -*-
# @Time    : 10/14/2020 11:57 AM
# @Author  : Chloe Ouyang
# @FileName: cnn.py.py
import tensorflow as tf


class CNN(tf.keras.layers.Layer):
    def call(self, input, vocab_size, feature_size, embed_size, num_classes, num_filters, filter_sizes):
        embed_initer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)




