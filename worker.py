from torch.multiprocessing import Process
import numpy as np

from wrapper import make_env

class Worker(Process):
    def __init__(self, idx, env_name, queue, barrier, state, reward, finished, score_channel):
        """mean: np.array of shape equal to the env state shape
        std: scalar, std of the obs collected by the random agent
        """
        super(Worker, self).__init__()
        self.daemon = True
        self.idx = idx
        self.env_name = env_name

        self.queue = queue
        self.barrier = barrier

        self.state = state
        self.reward = reward
        self.finished = finished

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
                self.state[self.idx] = state
                self.barrier.put(True)
            else:
                step += 1
                state, reward, done, _ = self.env.step(action)
                score += reward
                if done:
                    state = self.env.reset()
                    self.score_channel.put((score, step))
                    score = 0
                    step = 0
                
                self.state[self.idx] = state
                self.reward[self.idx] = reward
                self.finished[self.idx] = done

                self.barrier.put(True)