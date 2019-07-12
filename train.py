import torch
import numpy as np
import scipy
from torch.multiprocessing import Queue, RawArray
from ctypes import c_float
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
        self.barrier = Queue()  # This is used as a waiting mechanism, to wait for all the agents to env.step()
        self.score_channel = Queue()

        # these are shmem np.arrays
        self.state, self.reward, self.finished = self.init_shared()

        self.workers = [
            Worker(i, args.env, self.queues[i], self.barrier, self.state, self.reward, self.finished, self.score_channel) for i in range(self.num_actors)
        ]
        self.start_workers()
        
        self.model  = Policy(self.c_in, self.num_actions).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.eta)

        # used for logging and graphing
        self.stat = {
            'scores': [],
            'steps': [],
            'clip_losses': [],
            'value_losses': [],
            'entropies': []
        }

    def init_shared(self):
        state_shape = (self.num_actors, *self.obs_shape)
        scalar_shape = (self.num_actors, 1)

        state = np.empty(state_shape, dtype=np.float32)
        state = RawArray(c_float, state.reshape(-1))
        state = np.frombuffer(state, c_float).reshape(state_shape)

        reward = np.empty(scalar_shape, dtype=np.float32)
        reward = RawArray(c_float, reward.reshape(-1))
        reward = np.frombuffer(reward, c_float).reshape(scalar_shape)

        finished = np.empty(scalar_shape, dtype=np.float32)
        finished = RawArray(c_float, finished.reshape(-1))
        finished = np.frombuffer(finished, c_float).reshape(scalar_shape)

        return state, reward, finished
    
    def start_workers(self):
        for worker in self.workers:
            worker.start()

    def initialize_state(self):
        for i in range(self.num_actors):
            self.queues[i].put(-1)
        self.wait_for_agents()

    @timing_wrapper
    def broadcast_actions(self, actions):
        actions = actions.cpu().numpy()
        for i in range(self.num_actors):
            self.queues[i].put(actions[i])
        self.wait_for_agents()

        next_state = torch.tensor(self.state).to(self.device)
        reward = torch.tensor(self.reward).to(self.device)
        done = torch.tensor(self.finished).to(self.device)
        return next_state, reward, done
    
    def wait_for_agents(self):
        for i in range(self.num_actors):
            self.barrier.get()
    
    def setup_scheduler(self, T_max):
        num_steps = T_max // (self.horizon * self.num_actors)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lambda x: max(1 - x / num_steps, 0))

    @timing_wrapper
    def train(self, T_max, graph_name=None):
        self.setup_scheduler(T_max)

        global_step = 0

        self.initialize_state()
        state = torch.tensor(self.state).to(self.device)
        while global_step < T_max:

            states      = []
            actions     = []
            rewards     = []
            finished    = []
            sampled_lps = []  # sampled log probabilities
            values      = []

            time_start = time.time()
            duration_fwd = 0
            with torch.no_grad():
                for t in range(self.horizon):
                    global_step += self.num_actors

                    logit, value = self.model(state)
                    prob = torch.softmax(logit, dim=1)
                    log_prob = torch.log_softmax(logit, dim=1)

                    action = prob.multinomial(1)
                    sampled_lp = log_prob.gather(1, action)

                    (next_state, reward, done), duration_brdcst = self.broadcast_actions(action)

                    # appending to buffer
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    finished.append(done)
                    sampled_lps.append(sampled_lp)
                    values.append(value)

                    state = next_state

                    duration_fwd += duration_brdcst

                _, V = self.model(next_state)
                values.append(V)

            time_forward = time.time()

            # GAE estimation
            GAEs, duration_GAE = self.compute_GAE(rewards, finished, values)
            
            duration_backward = self.run_gradient_descent(states, actions, sampled_lps, values, GAEs)

            time_end = time.time()

            total_duration = time_end - time_start
            percent_broadcast = duration_fwd / total_duration * 100
            percent_forward = (time_forward - time_start) / total_duration * 100
            percent_GAE = duration_GAE / total_duration * 100
            percent_backward = duration_backward / total_duration * 100

            # print(f"<Time> Total: {total_duration:.2f} | forward: {percent_forward:.2f}% (broadcast {percent_broadcast:.2f}%) | GAE: {percent_GAE:.2f}% | backward: {percent_backward:.2f}%")
            if global_step % (self.num_actors * self.horizon * 30) == 0:
                while not self.score_channel.empty():
                    score, step = self.score_channel.get()
                    self.stat['scores'].append(score)
                    self.stat['steps'].append(step)
                now = datetime.datetime.now().strftime("%H:%M")
                print(f"Step {global_step} | Mean of last 10 scores: {np.mean(self.stat['scores'][-10:]):.2f} | Time: {now}")
                if graph_name is not None:
                    plot(global_step, self.stat, graph_name)
        # Finish
        plot(global_step, self.stat, graph_name)

    @timing_wrapper
    def compute_GAE(self, rewards, finished, values):
        GAEs = []
        advantage = 0
        for i in reversed(range(self.horizon)):
            td_error = rewards[i] + (1 - finished[i]) * self.gamma * values[i + 1] - values[i]
            advantage = td_error + (1 - finished[i]) * self.gamma * self.lam * advantage
            GAEs.append(advantage)
        GAEs = torch.cat(GAEs[::-1]).to(self.device)

        # NOTE: Below is currently not in use because I don't know how to incorporate the 'finished' tensor into account
        # NOTE: This version is much, much faster than the python-looped version above
        # NOTE: But in terms of the total time taken, it doesn't make much of a difference. (~2% compared to ~0.05%)
        # rewards = torch.stack(rewards)
        # finished = torch.stack(finished)
        # values = torch.stack(values)
        
        # td_error = rewards + (1 - finished) * self.gamma * values[1:] - values[:-1]
        # td_error = td_error.cpu()

        # GAEs = scipy.signal.lfilter([1], [1, -self.gamma * self.lam], td_error.flip(dims=(0,)), axis=0)
        # GAEs = np.flip(GAEs, axis=0)  # flip it back again
        # GAEs = GAEs.reshape(-1, GAEs.shape[-1])  # (horizon, num_actors, 1) --> (horizon * num_actors, 1)
        # GAEs = torch.tensor(GAEs).float().to(self.device)

        return GAEs

    @timing_wrapper
    def run_gradient_descent(self, states, actions, sampled_lps, values, GAEs):

        states      = torch.cat(states)
        actions     = torch.cat(actions)
        sampled_lps = torch.cat(sampled_lps)
        values      = torch.cat(values[:-1])
        targets     = GAEs + values

        self.scheduler.step()
        # Running SGD for K epochs
        for it in range(self.num_iter):
            # Batch indices
            idx = np.random.randint(0, self.horizon * self.num_actors, self.batch_size)
            
            state      = states[idx]
            action     = actions[idx]
            sampled_lp = sampled_lps[idx]
            GAE        = GAEs[idx]
            value      = values[idx]
            target     = targets[idx]

            # Normalize advantages
            GAE = (GAE - GAE.mean()) / (GAE.std() + 1e-8)

            logit_new, value_new = self.model(state)
            # Clipped values are needed because sometimes values can unexpectedly get really big
            clipped_value_new = value + torch.clamp(value_new - value, -self.eps, self.eps)

            # Calculating policy loss
            prob_new = torch.softmax(logit_new, dim=1)
            lp_new = torch.log_softmax(logit_new, dim=1)
            entropy = -(prob_new * lp_new).sum(1).mean()

            sampled_lp_new = lp_new.gather(1, action)

            ratio = torch.exp(sampled_lp_new - sampled_lp)
            surr1 = ratio * GAE
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * GAE
            clip_loss = torch.min(surr1, surr2).mean()

            # Calculating value loss
            value_loss1 = (value_new - target).pow(2)
            value_loss2 = (clipped_value_new - target).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

            final_loss = -clip_loss + value_loss - 0.01 * entropy

            self.optim.zero_grad()
            final_loss.backward()
            
            # total_norm = 0
            # for p in self.model.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1. / 2)
            # print(total_norm)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optim.step()

            # graphing
            self.stat['clip_losses'].append(clip_loss.item())
            self.stat['value_losses'].append(value_loss.item())
            self.stat['entropies'].append(entropy.item())




        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help="Name of the environment to use")
    parser.add_argument('--graph', default='graph.png', type=str, help="Name of the graph (must append .png, .jpeg etc)")

    parser.add_argument('--Tmax', default=1e+7, type=float, help="Max number of frames. Default is 10 million")
    parser.add_argument('--horizon', default=128, type=int, help="Horizon (T)")
    parser.add_argument('--eta', default=2.5*1e-4, type=float, help="Learning rate")
    parser.add_argument('--epoch', default=3, type=int, help="K, the number of epochs to run SGD on")
    parser.add_argument('--batch', default=32, type=int, help="Batch size per actor. If batch is K and there are N actors, the total batch size is NK")
    parser.add_argument('--gamma', default=0.99, type=float, help="Gamma (for discount)")
    parser.add_argument('--lam', default=0.95, type=float, help="Lambda (for GAE)")
    parser.add_argument('--actors', default=8, type=int, help="Number of actors")
    parser.add_argument('--eps', default=0.2, type=float, help="Eps, the clipping parameter")
    parser.add_argument('--save', default="", type=str, help="Location to save the model. Default: <None>")
    
    args = parser.parse_args()

    print(f"""<Configuration>
    Environment:      {args.env}
    Graph:            {args.graph}
    Max frames:       {args.Tmax}
    Horizon (T):      {args.horizon}
    Learning rate:    {args.eta}
    Epoch (optim):    {args.epoch}
    Total Batch size: {args.batch}x{args.actors}={args.batch * args.actors}
    Gamma (discount): {args.gamma}
    Lambda (GAE):     {args.lam}
    Num actors:       {args.actors}
    Epsilon (clip):   {args.eps}
    Current time:     {datetime.datetime.now()}
    Save location:    {args.save if args.save else "-"}
    """)

    trainer = PPOTrainer(args=args)
    time_taken = trainer.train(args.Tmax, args.graph)
    
    hms = str(datetime.timedelta(seconds=time_taken)).split(':')

    if args.save != "":
        torch.save(trainer.model, args.save)

    print(f"Total time taken: {hms[0]} hours {hms[1]} minutes (={int(time_taken)} seconds)")