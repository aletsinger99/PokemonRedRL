from gymnasium import Env
from gym import spaces
from pyboy.utils import WindowEvent
from pyboy import PyBoy
from memoryAddresses import *
import numpy as np
import time
from copy import deepcopy

import matplotlib.image
from pathlib import Path
import os

class RedEnv(Env):
    
    def __init__(self, window='null', ROM='ROM/PokemonRed.gb', initial_state_file="ROM/PokemonRed.gb.state", sparse_rewards=False):
        
        self.ROM = ROM
        self.window = window
        self.pyboy = PyBoy(self.ROM, window=self.window)
        self.pyboy.set_emulation_speed(0)
        self.initial_state_file = initial_state_file
        

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START
        ]
        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]
        
        # Define the state vector
         
        self.x_pos = 0
        self.y_pos = 0
        self.map_loc = 0
        self.type_of_battle = 0
        self.slot1 = 0
        self.slot1_hp = 0
        self.enemy_mon = 0
        self.enemy_mon_hp = 0
        self.party_levels = [0,0,0,0,0,0]
        self.party = [0,0,0,0,0]
        self.party_hp = [0,0,0,0,0,0]
        self.party_max_hp = [0,0,0,0,0,0]
        self.gym_badges = [0,0,0,0,0,0,0,0]
        self.flags = 0
        self.prev_flags = 0
        self.new_pos = 0
        self.seen_position = {}
        self.seen_location = {}
        self.state = np.hstack([self.x_pos, self.y_pos, self.map_loc, self.type_of_battle, self.slot1, self.slot1_hp, self.enemy_mon, self.enemy_mon_hp, self.party_levels, self.flags])
        # print(self.state)
        self.observation_space = spaces.Box(low=0, high = np.inf, shape = (len(self.state),), dtype=int)
        self.reward = 0

        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.sparse_rewards = sparse_rewards
        self.reset()
        # self.pyboy = PyBoy('ROM/PokemonRed.gb')

    def step(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # time.sleep(1/60)
        # disable rendering when we don't need it
        # action=action
        for i in range(3):
            self.pyboy.tick()
        match action:
            case 0:
                self.pyboy.send_input(self.release_arrow[0])
            case 1:
                self.pyboy.send_input(self.release_arrow[1])
            case 2:
                self.pyboy.send_input(self.release_arrow[2])
            case 3:
                self.pyboy.send_input(self.release_arrow[3])
            case 4:
                self.pyboy.send_input(self.release_button[0])
            case 5:
                self.pyboy.send_input(self.release_button[1])
            case 6:
                self.pyboy.send_input(self.release_button[2])
            case _:
                pass
        for j in range(0, 8):
            self.pyboy.tick()
        # self.pyboy.tick()
        self.update_state()
        self.update_reward()
        
        return self.state, self.reward, False, {} 
    def observe(self):
        return self.state

    def reset(self):
        self.pyboy.load_state(open(self.initial_state_file,"rb"))
        self.seen_position = {}
        self.seen_location = {}
        self.reward_level = None
        return self.observe()

    def read_m(self, addr):
        # return self.pyboy.get_memory_value(addr)
        return self.pyboy.memory[addr]

    def get_event_flags(self):
        bitsum = 0
        for i in range(55111,55431):
            bitsum = bitsum + ((self.read_m(i)).bit_count())
        self.prev_flags = self.flags
        self.flags = bitsum
      
            
    def get_badges(self):
        self.gym_badges = [self.read_m(BROCK), self.read_m(MISTY), self.read_m(SURGE),self.read_m(ERIKA), self.read_m(KOGA), self.read_m(BLAINE), self.read_m(SABRINA), self.read_m(GIOVANNI)]
        

    def get_position(self):
        self.x_pos = self.read_m(X_POS_ADDRESS)
        self.y_pos = self.read_m(Y_POS_ADDRESS)
        self.map_loc = self.read_m(MAP_N_ADDRESS)
        key=str(self.map_loc)+str(X_POS_ADDRESS)+str(Y_POS_ADDRESS)
        loc=str(self.map_loc)
        if (key in self.seen_position):
            self.seen_position[key] += 1
            self.new_pos= .99*self.seen_position[key]
        else:
            self.seen_position[key] = 1
            self.new_pos = 5
        if(loc not in self.seen_location):
            self.seen_location[loc]=1
            self.new_pos = 100

    def get_party(self):
        self.party = [self.read_m(a) for a in PARTY_ADDRESSES]
        self.party_levels = [self.read_m(a) for a in LEVELS_ADDRESSES]
        self.party_hp = [self.read_m(a) for a in HP_ADDRESSES]
        self.party_max_hp = [self.read_m(a) for a in MAX_HP_ADDRESSES]
    
    def get_battle(self):
        self.type_of_battle = self.read_m(CAN_CATCH)
        self.slot1 = self.read_m(POKEMON_1)
        self.slot1_hp = int(str(bin(self.read_m(POKEMON_1_H1)))[2:]+str(bin(self.read_m(POKEMON_1_H2)))[2:],2)
        self.enemy_mon = self.read_m(E_POKEMON)
        self.enemy_mon_hp = int(str(bin(self.read_m(E_POKEMON_H1)))[2:]+str(bin(self.read_m(E_POKEMON_H2)))[2:],2)

    def update_state(self):
        self.get_event_flags()
        self.get_badges()
        self.get_position()
        self.get_party()
        self.get_battle()
        
        self.state = np.hstack([self.x_pos, self.y_pos, self.map_loc, self.type_of_battle, self.slot1, self.slot1_hp, self.enemy_mon, self.enemy_mon_hp, self.party_levels,self.flags])
        # print(self.party_max_hp)
        # print(self.gym_badges)
        # print(self.state)
    
    def instant_reward(self):

        self.reward_level = self.flags + .1*len(self.seen_location) + np.sum(self.party_levels)

    def update_reward(self):
        
        # save old reward
        self.old_reward = self.reward_level
        self.instant_reward()

        # If you've progressed, maybe save your state
        if not (self.old_reward is None):
            if self.reward_level != self.old_reward:
                self.check_state_save()

        if self.sparse_rewards:
            if self.old_reward is None:
                self.reward = 0
            else:
                self.reward = self.reward_level - self.old_reward
        else:
            self.reward = self.reward_level

    def check_state_save(self, savedir="saved_states"):
        saves = os.listdir(savedir)
        savename = f"{int(self.flags)}_{len(self.seen_location)}_{int(np.sum(self.party_levels))}.state"
        savename = str(Path(savedir, savename))
        if saves:
            stats = [x.replace(".state", "").split("_") for x in saves if ".state" in x]
            max_flag = max([s[0] for s in stats])
            max_loc = max([s[1] for s in stats])
            max_level = max([s[2] for s in stats])
            if self.flags > int(max_flag) or len(self.seen_location) > int(max_loc) or np.sum(self.party_levels) > int(max_level):
                self.save_state(savename)
                self.save_screenshot(savename.replace(".state", ".png"))
        else:
            self.save_state(savename)
            self.save_screenshot(savename.replace(".state", ".png"))

    def save_state(self, file):
        with open(file, "wb") as f:
            self.pyboy.save_state(f)

    def load_state(self, file):
        self.pyboy.load_state(open(file,"rb"))

    def save_screenshot(self, file):
        rgba = self.pyboy.screen.ndarray
        matplotlib.image.imsave(file, rgba)

        # self.reward = self.flags

        # if self.reward is None:
        #     self.reward = 0
        

    
        
        


