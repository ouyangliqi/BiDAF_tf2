import tensorflow as tf


class C2QAttention(tf.keras.layers.Layer):

    def call(self, similarity, qencode):
        # homework
        c2q_att = tf.matmul(tf.nn.softmax(similarity, -1), qencode)

        return c2q_att


class Q2CAttention(tf.keras.layers.Layer):

    def call(self, similarity, cencode):
        # homework

        b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(similarity, 2), 1), -1)
        q2c_att = tf.tile(tf.matmul(b, cencode),
                          [1, tf.shape(cencode)[1], 1])

        return q2c_att
