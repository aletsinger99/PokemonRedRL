import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import RedEnvironment

# Instantiate the current environment as the imported Pokemon Red Gym Environment
env = RedEnvironment.RedEnv()
# state_size = env.observation_space.length
state_size = 41
actions_size = env.action_space.n
print(state_size)
print(actions_size)
class Q_est(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.hidden1 = nn.Linear(n_observations, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, n_actions)

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
    obs = np.array(s).astype('float')
    print(obs)
    if random.random() < eps:
        return np.argmax(Q(torch.tensor(s)).detach().numpy())
    # check if this is inclusive
    return random.randint(1,7)

Q = Q_est(state_size, actions_size)
Qt = Q_est(state_size, actions_size)
QBest = Q_est(state_size, actions_size)

def loss_fn(exptup, Q, Qt, discount):
    if not exptup.done:
        return (exptup.r + discount * torch.max(Qt(torch.tensor(exptup.sp)))-Q(torch.tensor(exptup.s))[exptup.a])**2
    else:
        return (exptup.r-Q(torch.tensor(exptup.s))[exptup.a]**2)


num_eps = 100
losses = []
buffer = []
epsilon = .99
max_steps = 500
optimizer = optim.Adam(Q.parameters(), lr=0.001)
discount = .999
for j in range(1, num_eps):
    
    env.reset()
    done = False
    Qt = Q
    step = 0
    while not done:
        if step > 3000:
            break
        s = env.observe()
        action = epsilon_policy(Q,s,epsilon)
        sp, r, done, temp = env.step(action)
        if done:
            print('We beat Pokemon Red, somehow....')
        step += 1
        experience_tuple = ObservationTuple(s, action, r, sp, done)
        buffer.append(experience_tuple)

        optimizer.zero_grad()
        random.shuffle(buffer)
        r_loss = 0
        data = buffer[0]
        pred = Q(torch.tensor(data.s))
        loss = loss_fn(data, Q, Qt, discount)
        loss.backward()
        optimizer.step()
        r_loss = r_loss + loss.detach().numpy()
        losses.append(r_loss)
        if step % 1000 == 0:
            print(f'Finished step {step}, latest loss {r_loss}')
            Qt = Q
            buffer = []
            r_loss = 0