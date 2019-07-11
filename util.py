import time
import torch, numpy as np
import matplotlib.pyplot as plt

def timing_wrapper(func):
    def wrapper(*args, **kwargs):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        return (t2 - t1), res
    return wrapper

def plot(cur_step, stat, name):
    plt.figure(figsize=(20,10))

    plt.subplot(231)
    last_10 = stat['scores'][-10:]
    plt.title(f'Running avg. score: {np.mean(last_10):.2f}, Frame: {cur_step}')
    plt.xlabel('# episodes')
    plt.ylabel('score')
    plt.plot(stat['scores'])

    plt.subplot(232)
    last_10k = stat['clip_losses'][-10000:]
    plt.title(f'clip_loss: {np.mean(last_10k):.4e} (mean of 10K)')
    plt.xlabel('# frames')
    plt.ylabel('clip_loss')
    plt.plot(stat['clip_losses'])

    plt.subplot(233)
    last_10k = stat['value_losses'][-10000:]
    plt.title(f'vloss: {np.mean(last_10k):.4e} (mean of 10K)')
    plt.xlabel('# frames')
    plt.ylabel('vloss')
    plt.plot(stat['value_losses'])

    plt.subplot(234)
    last_10k = stat['entropies'][-10000:]
    plt.title(f'entropy: {np.mean(last_10k):.4f} (mean of 10K)')
    plt.xlabel('# frames')
    plt.ylabel('entropy')
    plt.plot(stat['entropies'])

    plt.subplot(235)
    last_10k = stat['steps'][-10000:]
    plt.title(f'step: {np.mean(last_10k):.4f} (mean of 10K)')
    plt.xlabel('# episodes')
    plt.ylabel('step')
    plt.plot(stat['steps'])

    plt.savefig(name)
    plt.close()