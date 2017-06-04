import tensorflow as tf
import tensorflow.contrib.slim as slim


class QNetwork:
    def __init__(self, input_shape=(None, 210, 160, 3), num_out=6):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 210, 160, 3])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4],
                                 padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
                                 biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=128, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                 biases_initializer=None)

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        # self.streamAC, self.streamVC = tf.split(self.conv3, 2, 3)
        # self.streamA = slim.flatten(self.streamAC)
        # self.streamV = slim.flatten(self.streamVC)
        self.out_stream = slim.flatten(self.conv3)
        xavier_init = tf.contrib.layers.xavier_initializer()
        # self.AW = tf.Variable(xavier_init([self.streamA.shape[1], num_out]))
        # self.VW = tf.Variable(xavier_init([self.streamV.shape[1], 1]))
        self.AW = tf.Variable(xavier_init([self.out_stream.shape[1].value, num_out]))
        self.VW = tf.Variable(xavier_init([self.out_stream.shape[1].value, 1]))
        # self.Advantage = tf.matmul(self.streamA, self.AW)
        # self.Value = tf.matmul(self.streamV, self.VW)
        self.Advantage = tf.matmul(self.out_stream, self.AW)
        self.Value = tf.matmul(self.out_stream, self.VW)
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_out, dtype=tf.float32)

        self.Q = tf.reduce_sum(self.Qout * self.actions_onehot, axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class SimpleQNetwork:
    def __init__(self, input_shape=(210, 160, 3), num_out=6):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=(None, ) + input_shape, dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=(-1, ) + input_shape)
        self.conv1 = slim.conv2d(self.imageIn, num_outputs=32, kernel_size=(8, 8), stride=(4, 4), padding='VALID',
                                 biases_initializer=None)
        self.conv2 = slim.conv2d(self.conv1, num_outputs=64, kernel_size=(4, 4), stride=(2, 2), padding='VALID',
                                 biases_initializer=None)
        self.conv3 = slim.conv2d(self.conv2, num_outputs=128, kernel_size=(3, 3), stride=(1, 1), padding='VALID',
                                 biases_initializer=None)

        convsize = (self.conv3.shape.dims[1] * self.conv3.shape.dims[2] * self.conv3.shape.dims[3]).value
        self.res = tf.reshape(self.conv3, shape=(-1, convsize))
        self.dense1 = slim.relu(self.res, 512)
        self.drop1 = slim.dropout(self.dense1)
        self.dense3 = slim.relu(self.drop1, num_out)

        self.Qout = self.dense3
        self.predict = tf.arg_max(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_out, dtype=tf.float32)

        self.Q = tf.reduce_sum(self.Qout * self.actions_onehot, axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.updateModel = self.trainer.minimize(self.loss)
