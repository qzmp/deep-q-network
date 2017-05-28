import gym
import tensorflow as tf

from QNetwork import QNetwork, SimpleQNetwork
from utils import *

NUMBER_OF_EPISODES = 10000
PRE_TRAIN_STEPS = 100
BATCH_SIZE = 32
UPDATE_FREQUENCY = 4

env = gym.make('Pong-v0')
tf.reset_default_graph()
mainQN = SimpleQNetwork(num_out=env.action_space.n)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    episodeBuffer = ExperienceBuffer()
    for i in range(NUMBER_OF_EPISODES):
        state = env.reset()
        random_action = RandomActionChance()

        done = False
        total_steps = 0
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
            state_next, reward, done, unknown_dict = env.step(action)
            if reward != 0.0:
                reward_list.append(reward)
            episodeBuffer.add(np.array([state, action, reward, state_next]))
            total_steps += 1
            state = state_next
            # update network in intervals
            if total_steps >= PRE_TRAIN_STEPS and total_steps % UPDATE_FREQUENCY == 0:
                train_batch = episodeBuffer.sample(BATCH_SIZE)
                loss, _ = sess.run([mainQN.loss, mainQN.updateModel],
                                   feed_dict={mainQN.scalarInput: np.stack(train_batch[:, 0]),
                                              mainQN.targetQ: np.stack(train_batch[:, 2]),
                                              mainQN.actions: np.stack(train_batch[:, 1])})
                loss_list.append(loss)
            # end of episode
            if done:
                print(i, total_steps, np.mean(reward_list), np.mean(loss_list))
                break
