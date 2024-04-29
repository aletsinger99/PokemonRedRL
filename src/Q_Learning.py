import torch
import torch.nn as nn
import torch.optim as optim
import random
import RedEnvironment
import numpy as np
import argparse
import yaml

import time, sys
from pathlib import Path
import os
from copy import deepcopy

import matplotlib.pyplot as plt

class Q_est(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Q_est, self).__init__()
        self.hidden1 = nn.Linear(n_observations, 32)
        # self.hidden2 = nn.Linear(32, 16)
        self.output = nn.Linear(32, n_actions)


    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        # x = torch.relu(self.hidden2(x))
        # x = torch.sigmoid(self.hidden1(x))
        # x = self.hidden2(x)
        x = self.output(x)
        return x


class Q_est_ohe(nn.Module):
    def __init__(self, n_actions, n_feat = 2, n_loc = 256):
        super(Q_est_ohe, self).__init__()
        self.hidden1 = nn.Linear(n_feat + n_loc, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.output = nn.Linear(16, n_actions)


    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        # x = torch.sigmoid(self.hidden1(x))
        # x = self.hidden2(x)
        x = self.output(x)
        return x

    
class RNDist(nn.Module):
    def __init__(self, n_observations):
        super(RNDist, self).__init__()
        self.hidden1 = nn.Linear(n_observations, 32)
        self.output = nn.Linear(32, 1)


    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.sigmoid(self.output(x))
        return x
    

    
class RNDistOHE(nn.Module):
    def __init__(self, n_actions, n_feat = 2, n_loc = 256):
        super(RNDistOHE, self).__init__()
        self.hidden1 = nn.Linear(n_feat + n_loc, 16)
        self.hidden2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)


    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.sigmoid(self.output(x))
        return x


class ObservationTuple:
    def __init__(self, s, a, r, sp, done, s_ind = []):

        # Important info
        self.loc = int(s[3])
        self.type_of_battle = int(s[4])
        self.flags = s[-1]

        self.s = s
        if s_ind:
            self.s = self.s[s_ind]
        self.a = a
        self.r = r 
        self.sp = sp 
        if s_ind:
            self.sp = self.sp[s_ind]

        self.s = torch.tensor(self.s)
        self.sp = torch.tensor(self.sp)

        self.done = done


def encode_loc(s, s_ind, n_feat = 2, n_loc=256):
    
    new_s = np.zeros(n_feat + n_loc)
    new_s[:n_feat] = s[s_ind[:-1]]
    new_s[n_feat + int(s[s_ind[-1]])] = 1

    return torch.tensor(new_s, dtype=torch.float32).to_sparse()


def decode_loc(s, s_ind, n_feat = 2, n_loc=256):

    new_s = np.zeros(n_feat + 1)
    new_s[:n_feat] = s[:n_feat]
    new_s[n_feat] = np.argmax(s[n_feat:])

    return new_s


class ObservationTupleOHE:
    def __init__(self, s, a, r, sp, done, s_ind = []):

        # Important info
        self.loc = int(s[3])
        self.type_of_battle = int(s[4])
        self.flags = s[-1]

        self.s = encode_loc(s, s_ind)
        self.a = a
        self.r = r 
        self.sp = encode_loc(sp, s_ind)
        self.done = done




def epsilon_policy(Q, s, eps, s_ind=[], n_actions=5):
    if s_ind and len(s) < 100:
        s = s[s_ind]
        obs = np.array(s).astype('float32')
        obs = torch.tensor(obs)
    else:
        obs = s

    # print(obs)
    if random.random() > eps:
        # print(Q(obs))
        return np.argmax(Q(obs).detach().numpy())
    
    # check if this is inclusive
    return random.randint(0,n_actions-1)


def softmax_policy(Q, s, eps, s_ind=[], n_actions=5):
    if s_ind and len(s) < 100:
        s = s[s_ind]
        obs = np.array(s).astype('float32')
        obs = torch.tensor(obs)
    else:
        obs = s

    # print(obs)
    vals = Q(obs).detach().numpy()
    weights = np.exp(eps*vals)
    weights /= np.sum(weights)

    return np.random.choice(np.arange(n_actions), p = weights)



def loss_fn(exptup, Q, Qt, discount, intrinsic_reward):
    if not exptup.done:
        return (exptup.r + intrinsic_reward + discount * torch.max(Qt(torch.tensor(exptup.sp)))-Q(torch.tensor(exptup.s))[exptup.a])**2
    else:
        return (exptup.r + intrinsic_reward -Q(torch.tensor(exptup.s))[exptup.a]**2)

def loss_fn_mat(s, r, a, sp, done, Q, Qt, discount):
    Qsa = torch.flatten(torch.gather(Q(s), 1, a))
    return (1-torch.flatten(done))*(torch.flatten(r) + discount * torch.max(Qt(sp), dim=1)[0]-Qsa)**2 + torch.flatten(done)*(torch.flatten(r)-torch.flatten(Qsa)**2)

    
def tup_to_mat(data):
    s_mat = torch.tensor(np.vstack([b.s for b in data]))
    sp_mat = torch.tensor(np.vstack([b.sp for b in data]))
    a_mat = torch.tensor(np.vstack([b.a for b in data]))
    r_mat = torch.tensor(np.vstack([b.r for b in data]))
    done = torch.tensor(np.vstack([b.done for b in data]), dtype=torch.float32)
    return s_mat, r_mat, a_mat, sp_mat, done


def save_networks(folder, Q_dict, RND_pred_dict, RND_targ_dist):

    # Make the folder
    Path(folder).mkdir(exist_ok=True, parents=True)

    # Save all the dictionaries
    for loc in Q_dict:
        # Make subfolder
        subfolder = Path(folder, str(loc))
        subfolder.mkdir(exist_ok=True, parents=True)
        # Save dictionaries
        torch.save(Q_dict[loc][0].state_dict(), str(Path(subfolder, f"Q")))
        torch.save(RND_pred_dict[loc][0].state_dict(), str(Path(subfolder, f"RND_pred")))
        torch.save(RND_targ_dict[loc].state_dict(), str(Path(subfolder, f"RND_targ")))
    


def load_networks(folder, Q_func, Q_optim_func, RND_func, RND_optim_func):

    # Initialize dictionaires
    Q_dict, RND_pred_dict, RND_targ_dist = {}, {}, {}

    # List of locations
    locs = os.listdir(folder)

    # Load the networks
    for loc in locs:
        subfolder = Path(folder, loc)
        loc = int(loc)
        Q_dict[loc] = [Q_func(), None]
        Q_dict[loc][0].load_state_dict(torch.load(Path(subfolder, "Q")))
        Q_dict[loc][1] = Q_optim_func(Q_dict[loc][0])

        RND_pred_dict[loc] = [RND_func(), None]
        RND_pred_dict[loc][0].load_state_dict(torch.load(Path(subfolder, "RND_pred")))
        RND_pred_dict[loc][1] = RND_optim_func(RND_pred_dict[loc][0])

        RND_targ_dist[loc] = RND_func()
        RND_targ_dist[loc].load_state_dict(torch.load(Path(subfolder, "RND_targ")))

    return Q_dict, RND_pred_dict, RND_targ_dist



def evaluate_policy(env, Q_dict, max_steps, discount, state_idx, actions_size):
    
    env.reset()
    eps_reward = 0
    step = 0
    while step < max_steps:

        s = env.observe()
        loc = int(s[3])

        # Load networks
        Q = Q_dict[loc][0]

        # If you're not in a battle
        if s[4] == 0:
            action = epsilon_policy(Q, s, 0.1, state_idx, n_actions=actions_size)
        else:
            action = random.randint(0,5)
            
        sp, r, done, temp = env.step(action)

        eps_reward += r*(discount**step)
       
        step += 1
    
    return eps_reward



if __name__ == "__main__":

    # Open settings file
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--yaml_file", type=str, default="settings.yaml",
                        help="settings file")
    args = parser.parse_args()
    with open(args.yaml_file, 'r') as file:
        settings = yaml.safe_load(file)

    network_settings = settings["network"]
    num_eps = network_settings["num_eps"]
    epsilon = network_settings["epsilon"]
    max_steps = network_settings["max_step"]
    discount = network_settings["discount"]
    lr = network_settings["lr"]
    lr_rnd = network_settings["lr_rnd"]
    weights_file = network_settings["saved_weights"]
    save_root = network_settings["save_root"]
    train_every_n = network_settings["train_every_n"]
    save_every_n = network_settings["save_every_n"]
    batch_size = network_settings["batch_size"]
    best_state_file = network_settings["best_state_file"]
    running_screenshot = network_settings["running_screenshot"]
    reward_screenshots = network_settings["reward_screenshots"]
    initial_fill = network_settings["initial_fill"]

    print("-------------------------------------")
    print("Starting Q Learning with settings:")
    print(settings)
    print("-------------------------------------")


    # Instantiate the current environment as the imported Pokemon Red Gym Environment
    env = RedEnvironment.RedEnv(**settings["env"])
    # state_size = 41
    # state_idx = [1, 2, 3, -1] # X, Y, Location ID, Event Flag
    state_idx = [1, 2]
    state_size = len(state_idx)

    # actions_size = env.action_space.n
    # actions_size = 5 # down, left right, up, a
    actions_size = 4



    # Q = Q_est(state_size, actions_size)
    # Qt = Q_est(state_size, actions_size)
    # Q = Q_est_ohe(actions_size, state_size-1)
    # Qt = Q_est_ohe(actions_size, state_size-1)
    # QBest = Q_est(state_size, actions_size)

    # RNDistTarget = RNDist(state_size)
    # RNDistPredictor = RNDist(state_size)
    # RNDistTarget = RNDistOHE(actions_size, state_size-1)
    # RNDistPredictor = RNDistOHE(actions_size, state_size-1)

    Q_dict = {}
    RND_pred_dict = {}
    RND_targ_dict = {}
    Q_func = lambda : Q_est(state_size, actions_size)
    Q_optim_func = lambda Q: optim.Adam(Q.parameters(), lr=lr)
    RND_func = lambda : RNDist(state_size)
    RND_optim_func = lambda Q: optim.Adam(Q.parameters(), lr=lr_rnd)

    if not weights_file is None:
        Q_dict, RND_pred_dict, RND_targ_dict = load_networks(weights_file, Q_func, Q_optim_func,
                                                   RND_func, RND_optim_func)
        print("Loaded Weights")
    

    #     Q.load_state_dict(torch.load(weights_file))
    #     Qt.load_state_dict(torch.load(weights_file))

    losses = []
    buffer = []
    # optimizer = optim.Adam(Q.parameters(), lr=lr)
    # RNDoptimzer = optim.Adam(RNDistPredictor.parameters(), lr=1e-03)
    intrinsic_loss = torch.nn.MSELoss()
    t0 = time.time()
    epsMax = epsilon
    # epsRatio = 0.4/epsilon ** (1/num_eps)
    best_reward = 0
    eps_reward = 0
    eps_history = []
    count = 0
    max_prog = 0
    reward_history = []
    buff_size = 500000
    for j in range(1, num_eps+1):
        
        # reward and history for this episode
        env.reset()
        max_prog = 0
        eps_reward = 0
        eps_history = []
        
        # if epsilon > .1:
        #     epsilon = epsilon-.01
        epsilon = epsMax*(1-j/num_eps)
        # epsilon = epsilon

        done = False
        Qt_dict = deepcopy(Q_dict)
        step = 0
        while not done:
            if step > max_steps:
                break
            s = env.observe()
            # action = epsilon_policy(Q,s,epsilon, state_idx, n_actions=actions_size)
            # action = softmax_policy(Q,s,epsilon, state_idx, n_actions=actions_size)

            # Generate network for s if it doesn't exist
            loc = int(s[3])
            if not (loc in Q_dict):
                Q_dict[loc] = [Q_func(), None]
                Q_dict[loc][1] = Q_optim_func(Q_dict[loc][0])
                RND_pred_dict[loc] = [RND_func(), None]
                RND_pred_dict[loc][1] = RND_optim_func(RND_pred_dict[loc][0])
                RND_targ_dict[loc] = RND_func()
                print("Generated Networks for", loc)

            # Load networks
            Q = Q_dict[loc][0]
            optimizer = Q_dict[loc][1]
            RNDistPredictor = RND_pred_dict[loc][0]
            RNDoptimzer = RND_pred_dict[loc][1]
            RNDistTarget = RND_targ_dict[loc]

            # If you're not in a battle
            if s[4] == 0:
                action = epsilon_policy(Q, s, epsilon, state_idx, n_actions=actions_size)
                # action = softmax_policy(Q, encode_loc(s, state_idx),epsilon, state_idx, n_actions=actions_size)
                # action = epsilon_policy(Q, encode_loc(s, state_idx),epsilon, state_idx, n_actions=actions_size)
                # action = softmax_policy(Q, encode_loc(s, state_idx),epsilon, state_idx, n_actions=actions_size)
            else:
                action = random.randint(0,5)
                
            sp, r, done, temp = env.step(action)

            # Keep track of what the flags are
            prog = env.progress_val()
            if prog > max_prog:
                max_prog = prog
                env.save_screenshot(Path(reward_screenshots, f"{np.round(prog,2)}.png"))
            
            # if max_prog > network_settings["max_progress"]:
            #     done = True

            if sp[3] != s[3]:
                done = True

            step += 1
            count += 1

            experience_tuple = ObservationTuple(np.array(s).astype('float32'), action, r, np.array(sp).astype('float32'), done, state_idx)
            # experience_tuple = ObservationTupleOHE(np.array(s).astype('float32'), action, r, np.array(sp).astype('float32'), done, state_idx)
            buffer.append(experience_tuple)
            eps_history.append(experience_tuple)

            eps_reward += experience_tuple.r*(discount**step)

            if max_prog <= network_settings["max_progress"]:
                    done = False

            if done:
                print('We beat Pokemon Red, somehow....')

            if count % save_every_n == 0:
                # torch.save(Q.state_dict(), f"{save_root}_{count}")
                save_networks(f"{save_root}_{count}", Q_dict, RND_pred_dict, RND_targ_dict)

            if count % train_every_n == 0 and count >= initial_fill:

                env.save_screenshot(running_screenshot)
                env.visualize_policy("policy.png", Q, state_idx[2:])
                # env.visualize_policy("policy.png", Q, state_idx[2:], encode_fn=lambda x: encode_loc(x, state_idx))
                
                r_loss = 0
                data = np.random.choice(buffer, batch_size)
                rew = []
                intrin_loss_list = []
                for d in data:

                    # Select right networks
                    loc = d.loc
                    Q = Q_dict[loc][0]
                    if not (loc in Qt_dict):
                        Qt_dict[loc] = deepcopy(Q_dict[loc])
                    Qt = Qt_dict[loc][0]
                    optimizer = Q_dict[loc][1]
                    RNDistPredictor = RND_pred_dict[loc][0]
                    RNDoptimzer = RND_pred_dict[loc][1]
                    RNDistTarget = RND_targ_dict[loc]
                    
                    ir = 0

                    # if train_every_n > 1:

                    #     pred = Q(d.s)
                    #     intrinsicloss = intrinsic_loss(RNDistPredictor(d.s), RNDistTarget(d.s))
                    #     RNDoptimzer.zero_grad()
                        
                    #     intrin_loss_list.append(intrinsicloss.detach().numpy())
                    #     # print(np.std(intrin_std))
                    #     # intrinsic_reward = intrinsicloss/(np.std(intrin_loss_list)+.001)
                    #     intrinsic_reward = 100*intrinsicloss
                    #     intrinsicloss.backward()
                    #     RNDoptimzer.step()                    
                        
                    #     optimizer.zero_grad()
                    #     ir = intrinsic_reward.detach().numpy()
                    # else:
                    #     ir = 0

                    loss_Q = loss_fn(d, Q, Qt, discount, ir)
                    # loss_Q = loss_fn(d, Q, Qt, discount, 0)

                    # print(d.s.to_dense().numpy()[:2], str(ir))

                    loss_Q.backward()
                    optimizer.step()
                    
                    r_loss = r_loss + loss_Q.detach().numpy()
                    rew.append(d.r)
                if batch_size > 0:
                    losses.append(r_loss/batch_size)
                else:
                    losses.append(0)
                if train_every_n == 1:
                    time.sleep(0.1)
                # avg_rew.append(np.mean(rew))

                # breakpoint()
                # print(f'Finished step {step}, latest loss {r_loss}, Avgreward {np.mean(rew)}, Max Reward Achieved {np.max(rew)}')
                # print(f'Training: latest loss {r_loss}, Avgreward {np.mean(rew)}, Max Reward Achieved {np.max(rew)}')
                
                Qt_dict = deepcopy(Q_dict)
                if len(buffer) > buff_size:
                    # buffer = list(data)
                    buffer = buffer[buff_size - len(buffer):]
        
        print(f"Finished episode {j} -- episode reward {np.round(eps_reward, 3)} -- event flags seen {env.seen_event_flags}")
        print(env.seen_location)
        eval_rew = evaluate_policy(env, Q_dict, max_steps, discount,
                                   state_idx, actions_size)
        print(f"evaluated reward: {np.round(eval_rew, 3)}")
        if eval_rew >= best_reward:
            best_reward = eval_rew
            print("---------------------------")
            print(f"New best episode reward of {best_reward} achieved!")
            print("---------------------------")
            env.save_state(best_state_file)
            env.save_screenshot(best_state_file.replace(".state", ".png"))
            save_networks(f"{save_root}_best", Q_dict, RND_pred_dict, RND_targ_dict)

            # buffer = eps_history[:]
        
        eval_rew = evaluate_policy(env, Q_dict, max_steps=1000, 
                                   discount=discount,
                                   state_idx=state_idx,
                                   actions_size=actions_size)


        reward_history.append(eps_reward)
        plt.close()
        plt.plot(reward_history)
        plt.title("Reward vs. Episode #")
        plt.savefig("rewards.png")
        plt.close()

    tf = time.time()

    print("finished running in", np.round(tf-t0, 3), "seconds")