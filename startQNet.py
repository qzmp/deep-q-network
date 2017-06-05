import os

import gym
import tensorflow as tf

from QNetwork import SimpleQNetwork
from EnvWrapper import PongWrapper
from utils import *

NUMBER_OF_EPISODES = 10000
BATCH_SIZE = 32
UPDATE_FREQUENCY = 4

env = PongWrapper()
state = env.reset()
tf.reset_default_graph()
mainQN = SimpleQNetwork(input_shape=state.shape, num_out=env.action_space.n, dueling=True)
init = tf.global_variables_initializer()

# path = "logs/dueling"
tf.summary.scalar("loss", mainQN.loss)
tf.summary.scalar("acc", mainQN.acc)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/tensorflow_logs", graph=tf.get_default_graph())

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run(init)

    total_steps = 0
    episodeBuffer = ExperienceBuffer()
    random_action = RandomActionChance()
    for i in range(NUMBER_OF_EPISODES):
        state = env.reset()

        done = False
        loss_list = []
        reward_list = []
        while not done:
            # decide if do random action or network action
            if random_action.do_random_action() and total_steps == 0:
                action = env.action_space.sample()
                random_action.step_down()
            else:
                action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [state]})[0]
            # step action, actualize episode values
            state_next, reward, done = env.step(action)
            # if reward != 0.0:
            reward_list.append(reward)
            episodeBuffer.add(np.array([state, action, reward, state_next]))
            total_steps += 1
            state = state_next
            # update network in intervals
            if len(episodeBuffer) >= BATCH_SIZE and total_steps % UPDATE_FREQUENCY == 0:
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
                writer.add_summary(summary, total_steps)
                loss_list.append(loss)
            # end of episode
            if done:
                print(i, total_steps, np.mean([t for t in reward_list if t != 0.0]), np.mean(loss_list))
                break
