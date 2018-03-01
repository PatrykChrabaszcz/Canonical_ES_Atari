from src.policy import Policy
from natsort import natsorted as sorted
import numpy as np
import pickle
import gym
import os


class Visualizer(object):
    def __init__(self, game, network, train_directory):
        self.game = game
        env_name = '%sNoFrameskip-v4' % game
        env = gym.make(env_name)
        env = gym.wrappers.Monitor(env, '/tmp/temp_%s' % game, mode='evaluation', force=True)

        vb_file = os.path.join(train_directory, "vb.npy")
        vb = np.load(vb_file)
        parameters_file = sorted(os.listdir(train_directory))[-3]

        self.policy = Policy(env, network, "elu")

        parameters_path = os.path.join(train_directory, parameters_file)
        print('Using parameters file %s \n' % parameters_path)

        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)['params']

        self.policy.set_parameters(parameters)
        self.policy.set_vb(vb)

    def play_game(self):
        print(self.policy.rollout(render=True))


if __name__ == '__main__':
    vis = Visualizer('Qbert', 'Nature', train_directory='networks')
    vis.play_game()
