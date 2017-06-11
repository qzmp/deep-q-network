import tensorflow as tf
import tensorflow.contrib.slim as slim


class SimpleQNetwork:
    def __init__(self, input_shape=(210, 160, 3), num_out=6, dueling=False):
        self.scalarInput = tf.placeholder(shape=(None, ) + input_shape, dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=(-1, ) + input_shape)
        self.conv1 = slim.conv2d(self.imageIn, num_outputs=32, kernel_size=(8, 8), stride=(4, 4), padding='VALID',
                                 biases_initializer=None)
        self.conv2 = slim.conv2d(self.conv1, num_outputs=64, kernel_size=(4, 4), stride=(2, 2), padding='VALID',
                                 biases_initializer=None)
        self.conv3 = slim.conv2d(self.conv2, num_outputs=128, kernel_size=(3, 3), stride=(1, 1), padding='VALID',
                                 biases_initializer=None)

        self.flat = slim.flatten(self.conv3)
        if not dueling:
            self.dense1 = slim.relu(self.flat, 512)
            self.drop1 = slim.dropout(self.dense1)
            self.dense2 = slim.relu(self.drop1, num_out)
            self.Qout = self.dense2
        else:
            self.help_advantage = slim.relu(self.flat, 254)
            self.Advantage = slim.fully_connected(self.help_advantage, num_out)
            self.help_value = slim.relu(self.flat, 254)
            self.Value = slim.fully_connected(self.help_value, 1)
            self.Qout = self.Value + self.Advantage - tf.reduce_mean(self.Advantage, axis=1, keep_dims=True)

        self.predict = tf.arg_max(self.Qout, 1)

        # Values for learning
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int64)
        self.actions_onehot = tf.one_hot(self.actions, num_out, dtype=tf.float32)

        # Learning values
        self.Q = tf.reduce_sum(self.Qout * self.actions_onehot, axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdadeltaOptimizer(learning_rate=0.03)
        self.updateModel = self.trainer.minimize(self.loss)

        # For summary
        with tf.name_scope('Accuracy'):
            acc = tf.equal(self.predict, self.actions)
            self.acc = tf.reduce_mean(tf.cast(acc, tf.float32))
