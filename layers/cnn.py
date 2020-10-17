# -*- coding: utf-8 -*-
# @Time    : 10/14/2020 11:57 AM
# @Author  : Chloe Ouyang
# @FileName: cnn.py.py
import tensorflow as tf


class CNN(tf.keras.layers.Layer):
    def call(self, embed, feature_size, num_filters, filter_sizes):

        pool_outpus = []
        for filter_size in list(map(int, filter_sizes.split(','))):
            filter_shape = (filter_size, embed_size)
            conv = tf.keras.layers.Conv1D(num_filters, filter_shape, padding='valid',
                                          data_format='channels_last', activation='relu',
                                          kernel_initializer='glorot_normal',
                                          bias_initializer=tf.keras.initializers.constand(0.1),
                                          name='convolution_{:d'.format(filter_size))(embed)
            max_pool_shape = (feature_size - filter_size +1, 1)
            pool = tf.keras.layers.MaxPool1D(pool_size=max_pool_shape,
                                                 padding='valid',
                                                 data_format='channels_last',
                                                 name='max_pooling_{:d'.format(filter_size))(conv)
            pool_outpus.append(pool)


