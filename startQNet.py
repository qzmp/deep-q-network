import logging
import os
import sys

import tensorflow as tf

from QNetwork import SimpleQNetwork
from EnvWrapper import PongWrapper
from utils import *

NUMBER_OF_EPISODES = 10000
BATCH_SIZE = 32

env = PongWrapper()
state = env.reset()

tf.reset_default_graph()
if "dueling" in sys.argv:
    mainQN = SimpleQNetwork(input_shape=state.shape, num_out=env.action_space.n, dueling=True)
    path = "logs/dueling"
else:
    mainQN = SimpleQNetwork(input_shape=state.shape, num_out=env.action_space.n, dueling=False)
    path = "logs/simple"

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(os.path.join(path, "log.log")))
logger.setLevel(logging.INFO)

init = tf.global_variables_initializer()

tf.summary.scalar("loss", mainQN.loss)
tf.summary.scalar("acc", mainQN.acc)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(os.path.join(path, "tensorflow_logs"), graph=tf.get_default_graph())
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run(init)

    if tf.train.latest_checkpoint(path) is not None:
        saver.restore(sess, tf.train.latest_checkpoint(path))
        logger.info('Loading checkpoint')
    else:
        logger.info('Starting new learning')

    episodeBuffer = ExperienceBuffer()
    random_action = RandomActionChance()
    for i in range(NUMBER_OF_EPISODES):
        state = env.reset()

        done = False
        loss_list = []
        reward_list = []
        while not done:
            # decide if do random action or network action
            if random_action.do_random_action():
                action = env.action_space.sample()
                random_action.step_down()
            else:
                action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [state]})[0]
            # step action, actualize episode values
            state_next, reward, done = env.step(action)
            reward_list.append(reward)
            episodeBuffer.add(np.array([state, action, reward, state_next]))
            state = state_next
            # update network in intervals
            if len(episodeBuffer) >= BATCH_SIZE:
                train_batch = episodeBuffer.sample(BATCH_SIZE)
                predicted_reward = sess.run(mainQN.Qout, feed_dict={mainQN.scalarInput: np.stack(train_batch[:, 3])})
                rewards = []
                for j in range(len(train_batch)):
                    if train_batch[j][2] != 0.0:
                        rewards.append(train_batch[j][2])
                    else:
                        rewards.append(.9 * np.max(predicted_reward[j]))
                loss, _, summary = sess.run([mainQN.loss, mainQN.updateModel, merged],
                                            feed_dict={mainQN.scalarInput: np.stack(train_batch[:, 0]),
                                                       mainQN.actions: np.stack(train_batch[:, 1]),
                                                       mainQN.targetQ: rewards,
                                                       }
                                            )
                writer.add_summary(summary, i)
                loss_list.append(loss)
            # end of episode
            if done:
                logger.info(
                    ";".join([str(i), str(np.mean([t for t in reward_list if t != 0.0])), str(np.mean(loss_list))]))
                if i % 50 == 0:
                    saver.save(sess, os.path.join(path, 'model'), i)
