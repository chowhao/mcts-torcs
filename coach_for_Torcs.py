# from MCTS.MCTS import MCTS
from mcts.mcts import MCTS

from config.config import *
from utils.game_util import *
# from PIL import Image
# from io import BytesIO
# import base64
from network.avp import AvpNet
from network.vsp import VspNet
from game.game import Game
import tensorflow as tf
import numpy as np
# import time
# import os
from torcs.gym_torcs import TorcsEnv
# import cv2


class Coach:
    def __init__(self, game, avp_net):
        # self.train_interval = FLAGS.train_interval
        self.game = game
        self.avp_net = avp_net
        self.mcts = MCTS(self.game, self.avp_net)

        # store training dataset
        self.train_examples = []
        # 11 kinds of action
        self.action_num = FLAGS.action_num
        self.train_interval = FLAGS.train_interval

        self.step = 0
        self.last_state = None
        self.last_pi = 0
        self.last_a = 0


# session
sess = tf.Session()
avpNet = AvpNet(sess)
vspNet = VspNet(sess)
sess.run(tf.global_variables_initializer())
game_ = Game(sess, avpNet, vspNet)
coach = Coach(game_, avpNet)

# env
env = TorcsEnv(vision=True, throttle=False)
obs = env.reset()

steer_angle = 0.0
reward = 0.0
max_eps_steps = 4*1000
# max_eps_steps = 10000
episode_count = 2000

for i in range(episode_count):

    # relaunch TORCS every 3 episode because of the memory leak error
    if np.mod(i, 3) == 0:
        obs = env.reset(relaunch=True)
    else:
        obs = env.reset()

    for _ in range(max_eps_steps):

        coach.step += 1
        image = obs.img
        image = np.reshape(image, (64, 64, 3))

        image = process_img(image) / 255

        pos = (0, 0)
        #  a = "-90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90"
        # w is action
        w = coach.game.format_steer_angle(steer_angle)
        r = reward

        if coach.step > 1:
            coach.train_examples.append([coach.last_state, coach.last_pi, r, coach.last_a])
            # get enough data to train the net
            if len(coach.train_examples) >= coach.train_interval:
                # if len(coach.train_examples) <= coach.train_interval:
                coach.game.train_net(coach.train_examples)
                coach.train_examples = []

        # store current state waiting for next state to get reward
        state = np.reshape(image, (FLAGS.input_height, FLAGS.input_width, 1))
        coach.last_state = state

        coach.mcts = MCTS(coach.game, coach.avp_net)
        # 1 is tmp, pi is possibility
        pi = coach.mcts.get_action_prob(state, w, 1)

        action = np.argmax(pi)
        np.set_printoptions(precision=4)
        pi = np.array(pi)
        coach.last_a = action
        coach.last_pi = pi
        # steer_angle is [-1, 1]
        steer_angle = -1.0 + action / (coach.action_num - 1) * 2.0
        a_t = np.zeros((1,))
        a_t[0] = steer_angle

        obs, reward, done, _ = env.step(a_t)

        if reward < 0:
            reward = -1
        else:
            reward = 0.1
        print('eps', i, 'step', coach.step, 'r', r, 'action', a_t, 'pi', pi)
        if done:
            print('reset！' * 10)
            obs = env.reset()
