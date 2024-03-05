import gym
from gym import Env
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import RedEnvironment


env = RedEnvironment.RedEnv()
state = env.observation_space.shape
print(state)
# actions = env.valid_actions
actions = env.action_space.n
print(actions)


model = Sequential()
# model.add(InputLayer(input_shape=state))

model.add(Dense(41, activation="relu",input_shape=state))
model.add(Flatten())
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="tanh"))

Adam._name = ''
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions = actions,
    nb_steps_warmup=1,
    target_model_update=1
)

agent.compile(Adam(lr = 0.001), metrics=["mae"])
agent.fit(env, nb_steps=1000000, nb_max_episode_steps=50000) 
agent.save_weights("myweights", overwrite=True)