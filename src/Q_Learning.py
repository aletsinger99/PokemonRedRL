import torch
import torch.nn as nn
import torch.optim as optim
import random
import RedEnvironment
import numpy as np
import argparse
import yaml

import time
    

class Q_est(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Q_est, self).__init__()
        self.hidden1 = nn.Linear(n_observations, 64)
        # print(self.hidden1.weight.dtype)
        self.hidden2 = nn.Linear(64, 32)
        # print(self.hidden2.weight.dtype)
        self.output = nn.Linear(32, n_actions)
        # print(self.output.weight.dtype)

    def forward(self, x):
        # x = torch.tensor(x, dtype=float)
        # print(x)
        # print(type(x))
        x = torch.relu(self.hidden1(x))
        # print(x)
        x = torch.relu(self.hidden2(x))
        # print(x)
        x = self.output(x)
        # print(x)
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
    return random.randint(0,6)


def loss_fn(exptup, Q, Qt, discount):
    if not exptup.done:
        return (exptup.r + discount * torch.max(Qt(torch.tensor(exptup.sp)))-Q(torch.tensor(exptup.s))[exptup.a])**2
    else:
        return (exptup.r-Q(torch.tensor(exptup.s))[exptup.a]**2)

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
    every_n = network_settings["every_n"]
    batch_size = network_settings["batch_size"]

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

    if not weights_file is None:
        Q.load_state_dict(torch.load(weights_file))
        Qt.load_state_dict(torch.load(weights_file))

    losses = []
    buffer = []
    optimizer = optim.Adam(Q.parameters(), lr=lr)
    t0 = time.time()
    r_loss = 0
    for j in range(1, num_eps+1):
        env.reset()
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
            if done:
                print('We beat Pokemon Red, somehow....')
            step += 1
            experience_tuple = ObservationTuple(np.array(s).astype('float32'), action, r, np.array(sp).astype('float32'), done)
            buffer.append(experience_tuple)

            if step % every_n == 0:
                
                data = np.random.choice(buffer, batch_size)
                for d in data:
                    optimizer.zero_grad()
                    pred = Q(torch.tensor(d.s))
                    loss = loss_fn(d, Q, Qt, discount)
                    loss.backward()
                    optimizer.step()
                    r_loss = r_loss + loss.detach().numpy()
                    losses.append(r_loss/batch_size)

                # breakpoint()
                print(f'Finished step {step}, latest loss {r_loss}')
                Qt = Q
                buffer = []
                torch.save(Q.state_dict(), f"{save_root}_{step}")

    tf = time.time()

    print("finished running in", np.round(tf-t0, 3), "seconds")