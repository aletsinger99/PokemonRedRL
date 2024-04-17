import torch
import torch.nn as nn
import torch.optim as optim
import random
import RedEnvironment
import numpy as np
import argparse
import yaml

import time
from pathlib import Path
    

class Q_est(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Q_est, self).__init__()
        self.hidden1 = nn.Linear(n_observations, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.hidden3 = nn.Linear(32, 16)

        self.output = nn.Linear(16, n_actions)


    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x
    
class RNDist(nn.Module):
    def __init__(self, n_observations):
        super(RNDist, self).__init__()
        self.hidden1 = nn.Linear(n_observations, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 16)


    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x


class ObservationTuple:
    def __init__(self, s, a, r, sp, done):
        self.s = s
        self.a = a
        self.r = r 
        self.sp = sp 
        self.done = done

def epsilon_policy(Q, s, eps):
    obs = np.array(s).astype('float32')
    obs = torch.tensor(obs)

    # print(obs)
    if random.random() > eps:
        # print(Q(obs))
        return np.argmax(Q(obs).detach().numpy())
    
    # check if this is inclusive
    return random.randint(0,4)


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
    state_size = 41
    actions_size = env.action_space.n



    Q = Q_est(state_size, actions_size)
    Qt = Q_est(state_size, actions_size)
    # QBest = Q_est(state_size, actions_size)
    RNDistTarget = RNDist(state_size)
    RNDistPredictor = RNDist(state_size)
    
    if not weights_file is None:
        Q.load_state_dict(torch.load(weights_file))
        Qt.load_state_dict(torch.load(weights_file))

    losses = []
    buffer = []
    optimizer = optim.Adam(Q.parameters(), lr=lr)
    RNDoptimzer = optim.Adam(RNDistPredictor.parameters(), lr=lr)
    intrinsic_loss = torch.nn.MSELoss()
    t0 = time.time()
    epsMax = epsilon
    # epsRatio = 0.4/epsilon ** (1/num_eps)
    best_reward = 0
    eps_reward = 0
    eps_history = []
    count = 0
    max_prog = 0
    for j in range(1, num_eps+1):
        
        # reward and history for this episode
        env.reset()
        eps_reward = 0
        eps_history = []
        
        if epsilon > .1:
            epsilon = epsilon-.01
        # epsilon = epsMax*epsRatio**(j-1)
        # epsilon = epsilon
        done = False
        Qt = Q
        step = 0
        while not done:
            if step > max_steps:
                break
            s = env.observe()
            action = epsilon_policy(Q,s,epsilon)
            # print(action)
            sp, r, done, temp = env.step(action)
            # Keep track of what the flags are
            prog = env.progress_val()
            if prog > max_prog:
                max_prog = prog
                env.save_screenshot(Path(reward_screenshots, f"{np.round(r,2)}.png"))

            if done:
                print('We beat Pokemon Red, somehow....')
            step += 1
            count += 1

            experience_tuple = ObservationTuple(np.array(s).astype('float32'), action, r, np.array(sp).astype('float32'), done)
            buffer.append(experience_tuple)
            eps_history.append(experience_tuple)
            eps_reward += experience_tuple.r*(discount**step)

            if count % save_every_n == 0:
                torch.save(Q.state_dict(), f"{save_root}_{count}")

            if count % train_every_n == 0 and count >= initial_fill:
                
                r_loss = 0
                data = np.random.choice(buffer, batch_size)
                rew = []
                intrin_loss_list = []
                for d in data:
                    
                    
                    pred = Q(torch.tensor(d.s))
                    intrinsicloss = intrinsic_loss(RNDistPredictor(torch.tensor(d.s)), RNDistTarget(torch.tensor(d.s)))
                    RNDoptimzer.zero_grad()
                    
                    intrin_loss_list.append(intrinsicloss.detach().numpy())
                    # print(np.std(intrin_std))
                    intrinsic_reward = intrinsicloss/(np.std(intrin_loss_list)+.001)
                    intrinsicloss.backward()
                    RNDoptimzer.step()
                    
                    optimizer.zero_grad()
                    loss_Q = loss_fn(d, Q, Qt, discount, intrinsic_reward.detach().numpy())
                    loss_Q.backward()
                    optimizer.step()
                    
                    r_loss = r_loss + loss_Q.detach().numpy()
                    rew.append(d.r)
                losses.append(r_loss/batch_size)
                # avg_rew.append(np.mean(rew))

                # breakpoint()
                print(f'Finished step {step}, latest loss {r_loss}, Avgreward {np.mean(rew)}, Max Reward Achieved {np.max(rew)}')

                env.save_screenshot(running_screenshot)
                Qt = Q
                if len(buffer) > 5000000:
                    buffer = list(data)
        
        print(f"Finished episode {j} -- episode reward {np.round(eps_reward, 3)} -- event flags seen {env.seen_event_flags}")
        if eps_reward > best_reward:
            best_reward = eps_reward
            print("---------------------------")
            print(f"New best episode reward of {best_reward} achieved!")
            print("---------------------------")
            env.save_state(best_state_file)
            env.save_screenshot(best_state_file.replace(".state", ".png"))
            # buffer = eps_history[:]
    
    tf = time.time()

    print("finished running in", np.round(tf-t0, 3), "seconds")