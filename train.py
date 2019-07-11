import torch
import numpy as np
import scipy
from torch.multiprocessing import Queue
import time, datetime
import argparse

from network import Policy
from wrapper import make_env
from util import timing_wrapper, plot
from worker import Worker


class PPOTrainer:
    def __init__(self, args):
        tmp_env = make_env(args.env)
        self.obs_shape = tmp_env.observation_space.shape
        self.num_actions = tmp_env.action_space.n
        self.c_in = self.obs_shape[0]
        del tmp_env

        self.horizon    = args.horizon
        self.eta        = args.eta
        self.epoch      = args.epoch
        self.batch_size = args.batch * args.actors
        self.gamma      = args.gamma
        self.lam        = args.lam
        self.num_actors = args.actors
        self.eps        = args.eps
        self.num_iter   = (args.epoch * args.actors * args.horizon) // self.batch_size # how many times to run SGD on the buffer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.queues = [Queue() for i in range(self.num_actors)]
        self.channel = Queue() # envs send their stuff through this channel
        self.score_channel = Queue()

        self.workers = [
            Worker(i, args.env, self.queues[i], self.channel, self.score_channel) for i in range(self.num_actors)
        ]
        self.start_workers()
        
        self.model  = Policy(self.c_in, self.num_actions).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # used for logging and graphing
        self.stat = {
            'scores': [],
            'steps': [],
            'clip_losses': [],
            'value_losses': [],
            'entropies': []
        }
    
    def start_workers(self):
        for worker in self.workers:
            worker.start()

    def get_init_state(self):
        states = torch.empty(self.num_actors, *self.obs_shape).float().to(self.device)
        for q in self.queues:
            q.put(-1)

        for i in range(self.num_actors):
            idx, state = self.channel.get()
            states[idx] = torch.tensor(state)
        return states

    @timing_wrapper
    def broadcast_actions(self, actions):
        states = torch.empty(self.num_actors, *self.obs_shape).float().to(self.device)
        rewards = torch.empty(self.num_actors, 1).float().to(self.device)
        finished = torch.empty(self.num_actors, 1).float().to(self.device)

        for i in range(self.num_actors):
            self.queues[i].put(actions[i].item())  # actions is torch.tensor

        for _ in range(self.num_actors):
            idx, state, reward, done = self.channel.get()
            states[idx] = torch.tensor(state)
            rewards[idx] = reward
            finished[idx] = done

        return states, rewards, finished

    def train(self, T_max, graph_name=None):
        global_step = 0

        state = self.get_init_state().to(self.device)
        while global_step < T_max:

            states      = []
            actions     = []
            rewards     = []
            finished    = []
            sampled_lps = []  # sampled log probabilities
            values      = []

            time_start = time.time()
            fwd_broadcast = 0
            with torch.no_grad():
                for t in range(self.horizon):
                    global_step += self.num_actors

                    logit, value = self.model(state)
                    prob = torch.softmax(logit, dim=1)
                    log_prob = torch.log_softmax(logit, dim=1)

                    action = prob.multinomial(1)
                    sampled_lp = log_prob.gather(1, action)

                    time_broadcast, (next_state, reward, done) = self.broadcast_actions(action)

                    # appending to buffer
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    finished.append(done)
                    sampled_lps.append(sampled_lp)
                    values.append(value)

                    state = next_state

                    fwd_broadcast += time_broadcast

                _, V = self.model(next_state)
                values.append(V)

            time_forward = time.time()

            # GAE estimation
            time_GAE, GAEs = self.compute_GAE(rewards, finished, values)
            
            time_backward, _ = self.run_gradient_descent(states, actions, sampled_lps, values, GAEs)

            time_end = time.time()

            total_time = time_end - time_start
            percent_broadcast = fwd_broadcast / (time_forward - time_start) * 100
            percent_forward = (time_forward - time_start) / total_time * 100
            percent_GAE = time_GAE / total_time * 100
            percent_backward = time_backward / total_time * 100

            print(f"<Time> Total: {total_time:.2f} | forward: {percent_forward:.2f}% (of which broadcast takes {percent_broadcast:.2f}%) | GAE: {percent_GAE:.2f}% | backward: {percent_backward:.2f}%")
            if global_step % (self.num_actors * self.horizon * 30) == 0:
                while not self.score_channel.empty():
                    score, step = self.score_channel.get()
                    self.stat['scores'].append(score)
                    self.stat['steps'].append(step)
                now = datetime.datetime.now().strftime("%H:%M")
                print(f"Step {global_step} | Mean of last 10 scores: {np.mean(self.stat['scores'][-10:]):.2f} | Time: {now}")
                plot(global_step, self.stat, graph_name)

    @timing_wrapper
    def compute_GAE(self, rewards, finished, values):
        rewards = torch.stack(rewards)
        finished = torch.stack(finished)
        values = torch.stack(values)
        
        td_error = rewards + (1 - finished) * self.gamma * values[1:] - values[:-1]
        td_error = td_error.cpu()

        GAEs = scipy.signal.lfilter([1], [1, -self.gamma * self.lam], td_error.flip(dims=(0,)), axis=0)
        GAEs = np.flip(GAEs, axis=0)  # flip it back again
        GAEs = GAEs.reshape(-1, GAEs.shape[-1])  # (horizon, num_actors, 1) --> (horizon * num_actors, 1)
        
        return torch.tensor(GAEs).float().to(self.device)

    @timing_wrapper
    def run_gradient_descent(self, states, actions, sampled_lps, values, GAEs):
        states      = torch.cat(states)
        actions     = torch.cat(actions)
        sampled_lps = torch.cat(sampled_lps)
        values      = torch.cat(values[:-1])

        # Running SGD for K epochs
        for it in range(self.num_iter):
            # batch indices
            idx = np.random.randint(0, self.horizon * self.num_actors, self.batch_size)
            
            state = states[idx]
            action = actions[idx]
            sampled_lp = sampled_lps[idx]
            GAE = GAEs[idx]
            value = values[idx]

            logit_new, value_new = self.model(state)

            prob_new = torch.softmax(logit_new, dim=1)
            lp_new = torch.log_softmax(logit_new, dim=1)
            entropy = -(prob_new * lp_new).sum(1).mean()

            sampled_lp_new = lp_new.gather(1, action)

            ratio = torch.exp(sampled_lp_new - sampled_lp)
            surr1 = ratio * GAE
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * GAE
            clip_loss = torch.min(surr1, surr2).mean()

            target = GAE + value
            value_loss = 0.5 * (value_new - target).pow(2).mean()

            final_loss = -clip_loss + value_loss - 0.01 * entropy

            self.optim.zero_grad()
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optim.step()

            # graphing
            self.stat['clip_losses'].append(clip_loss.item())
            self.stat['value_losses'].append(value_loss.item())
            self.stat['entropies'].append(entropy.item())




        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help="Name of the environment to use")
    parser.add_argument('--name', default='graph.png', type=str, help="Name of the graph (must append .png, .jpeg etc)")

    parser.add_argument('--Tmax', default=int(1e+7), type=int, help="Max number of frames. Default is 10 million")
    parser.add_argument('--horizon', default=128, type=int, help="Horizon (T)")
    parser.add_argument('--eta', default=2.5*1e-4, type=float, help="Learning rate")
    parser.add_argument('--epoch', default=3, type=int, help="K, the number of epochs to run SGD on")
    parser.add_argument('--batch', default=8, type=int, help="Batch size per actor. If batch is K and there are N actors, the total batch size is NK")
    parser.add_argument('--gamma', default=0.99, type=float, help="Gamma (for discount)")
    parser.add_argument('--lam', default=0.95, type=float, help="Lambda (for GAE)")
    parser.add_argument('--actors', default=32, type=int, help="Number of actors")
    parser.add_argument('--eps', default=0.2, type=float, help="Eps, the clipping parameter")
    
    args = parser.parse_args()

    print(f"""<Configuration>
    Environment:      {args.env}
    Graph:            {args.name}
    Max frames:       {args.Tmax}
    Horizon (T):      {args.horizon}
    Learning rate:    {args.eta}
    Epoch (optim):    {args.epoch}
    Total Batch size: {args.batch}x{args.actors}={args.batch * args.actors}
    Gamma (discount): {args.gamma}
    Lambda (GAE):     {args.lam}
    Num actors:       {args.actors}
    Epsilon (clip):   {args.eps}
    """)

    trainer = PPOTrainer(args=args)
    trainer.train(args.Tmax, args.name)