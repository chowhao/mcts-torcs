# The VSP network predicts the future state of the vehicle based on a given control action
from config.config import *
# import matplotlib.pyplot as plt
from utils.vsp_util import *


class VspNet:
    def __init__(self, sess):
        self.sess = sess
        self.lr = FLAGS.vsp_lr
        self.batch_size = FLAGS.vsp_batch_size

        self.action_num = FLAGS.action_num
        self._build_net()

    def _build_net(self):
        # input: state, action; output: next state, s_: real next state
        self.s = tf.placeholder('float', shape=[None, FLAGS.input_height, FLAGS.input_width, 1])
        self.s_ = tf.placeholder('float', shape=[None, FLAGS.input_height, FLAGS.input_width, 1])
        self.a = tf.placeholder('float', shape=[None, 1])
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.feed_batch_size = tf.placeholder(tf.int64)

        with tf.variable_scope('vsp_net'):
            # conv1 = conv_layer_with_bn(self.s, [8, 8, self.s.get_shape().as_list()[3], 16], self.phase_train,
            conv1 = conv_layer_with_bn(self.s, [8, 8, 1, 16], self.phase_train,
                                       name="conv1")
            # pool1
            pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                              padding='SAME', name='pool1')
            # conv2
            conv2 = conv_layer_with_bn(pool1, [4, 4, 16, 16], self.phase_train, name="conv2")

            # pool2
            pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                                                              strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            # conv3
            conv3 = conv_layer_with_bn(pool2, [3, 3, 16, 16], self.phase_train, name="conv3")

            # pool3
            pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                                                              strides=[1, 2, 2, 1], padding='SAME', name='pool3')
            self.indi = pool3_indices

            # fully connected
            pool3_shape = pool3.get_shape().as_list()
            pool3_size = np.prod(pool3_shape[1:])
            pool3_fc = tf.reshape(pool3, [-1, pool3_size])
            pool3_concat_a = tf.concat([pool3_fc, self.a], axis=1)
            fc = tf.layers.dense(pool3_concat_a, pool3_size, activation=tf.nn.relu)
            fc = tf.reshape(fc, [-1, pool3_shape[1], pool3_shape[2], pool3_shape[3]])

            # up_sample according to the pool_indices
            # up_sample3 and conv_decode3
            up_sample3 = self.un_pool_with_argmax(fc, pool3_indices)
            conv_decode3 = conv_layer_with_bn(up_sample3, [3, 3, 16, 16], self.phase_train, False, name="conv_decode3")

            # up_sample2 and conv_decode2
            up_sample2 = self.un_pool_with_argmax(conv_decode3, pool2_indices)
            conv_decode2 = conv_layer_with_bn(up_sample2, [4, 4, 16, 16], self.phase_train, False, name="conv_decode2")

            # up_sample1 and conv_decode1
            up_sample1 = self.un_pool_with_argmax(conv_decode2, pool1_indices)
            self.output = conv_layer_with_bn(up_sample1, [8, 8, 16, 1], self.phase_train, False, name="conv_decode1")

            with tf.variable_scope('loss'):
                l1 = tf.reshape(self.output, [-1, np.prod(self.output.get_shape().as_list()[1:])])
                l2 = tf.reshape(self.s_, [-1, np.prod(self.s_.get_shape().as_list()[1:])])
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(l1, l2), axis=1))
            with tf.variable_scope('train'):
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vsp_net')
                optimizer = tf.train.RMSPropOptimizer(self.lr)
                self._train_op = optimizer.minimize(self.loss, var_list=train_vars)

    def un_pool_with_argmax(self, bottom, argmax, output_shape=None, name='max_un_pool_with_argmax'):
        with tf.name_scope(name):
            k_size = [1, 2, 2, 1]
            input_shape = bottom.get_shape().as_list()
            #  calculation new shape
            if output_shape is None:
                output_shape = [-1,
                                input_shape[1] * k_size[1],
                                input_shape[2] * k_size[2],
                                input_shape[3]]

            flat_input_size = np.prod(input_shape[1:])
            flat_output_size = np.prod(output_shape[1:])
            bottom_ = tf.reshape(bottom, [self.feed_batch_size * flat_input_size])
            argmax_ = tf.reshape(argmax, [-1, 1])

            ret = tf.scatter_nd(argmax_, bottom_, [self.feed_batch_size*flat_output_size])

            ret = tf.reshape(ret, output_shape)
            return ret

    # input: image, action; output: next image or state
    def predict(self, obs, a):
        obs = np.reshape(obs, (1, FLAGS.input_height, FLAGS.input_width, 1))
        a_array = np.zeros((1, 1))
        a_array[0, 0] = a
        s_ = self.sess.run([self.output], feed_dict={self.s: obs, self.a: a_array,
                                                     self.phase_train: False, self.feed_batch_size: 1})
        s_ = np.squeeze(s_)
        return s_

    # input: batches of image, action, real next image
    def learn(self, obs_batch, a_batch, s_batch):
        _, loss = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.s: obs_batch, self.a: a_batch,
                                self.s_: s_batch, self.phase_train: True,
                                           self.feed_batch_size: self.batch_size})
        return loss
