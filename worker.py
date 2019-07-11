from torch.multiprocessing import Process
import numpy as np

from wrapper import make_env

class Worker(Process):
    def __init__(self, idx, env_name, queue, channel, score_channel):
        """mean: np.array of shape equal to the env state shape
        std: scalar, std of the obs collected by the random agent
        """
        super(Worker, self).__init__()
        self.daemon = True
        self.idx = idx
        self.env_name = env_name

        self.queue = queue
        self.channel = channel
        self.score_channel = score_channel

    def run(self):
        self.env = make_env(self.env_name)

        score = 0
        step = 0
        while True:
            action = self.queue.get()
            if action is None:
                break
            elif action == -1: # reset
                state = self.env.reset()
                self.channel.put((self.idx, state))
            else:
                step += 1
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    next_state = self.env.reset()
                    self.score_channel.put((score, step))
                    score = 0
                    step = 0
                
                transition = (self.idx, next_state, reward, done)
                self.channel.put(transition)