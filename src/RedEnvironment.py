from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
from pyboy import PyBoy
from memoryAddresses import *
import numpy as np
import time
from itertools import product
from copy import deepcopy
import matplotlib.image
import matplotlib.pyplot as plt
import torch

class RedEnv(Env):
    
    def __init__(self, window='SDL2', ROM='ROM/PokemonRed.gb', initial_state_file="ROM/PokemonRed.gb.state"):
        
        self.ROM = ROM
        self.window = window
        self.pyboy = PyBoy(self.ROM, window=self.window)
        self.pyboy.set_emulation_speed(0)
        self.initial_state_file = initial_state_file
        self.reset()

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
        self.new_flags = 0
        self.new_pos = 0
        self.seen_position = {}
        self.seen_location = {}
        self.state = np.hstack([1, self.x_pos, self.y_pos, self.map_loc, self.type_of_battle, self.slot1, self.slot1_hp, self.enemy_mon, self.enemy_mon_hp, self.party_levels, self.party, self.party_hp, self.party_max_hp, self.gym_badges, self.flags]).reshape(41)
        self.observation_space = spaces.Box(low=0, high = 1000, shape = (1,len(self.state)), dtype=int)
        self.reward = 0

        self.action_space = spaces.Discrete(len(self.valid_actions))
        # self.pyboy = PyBoy('ROM/PokemonRed.gb')

        self.old_seen_loc = 0

    def step(self, action):

        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # time.sleep(1/60)
        # disable rendering when we don't need it
        # action=action
        for j in range(0, 4):
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
        
        self.update_state()
        self.update_reward()
        
        return self.state, self.reward, False, {} 
    def observe(self):
        return self.state.astype('float32')

    def reset(self):
        self.load_state(self.initial_state_file)
        self.seen_position = {}
        self.seen_event_flags = {}
        self.seen_location = {}
        self.old_seen_loc = 0
        self.last_action = None
        return
    def read_m(self, addr):
        # return self.pyboy.get_memory_value(addr)
        return self.pyboy.memory[addr]

    def get_event_flags(self):
        bitsum = 0
        for i in range(55111,55431):
            bitsum = bitsum + ((self.read_m(i)).bit_count())
        
        self.new_flags = bitsum - self.flags
        self.flags = bitsum

        if bitsum in self.seen_event_flags:
            self.seen_event_flags[bitsum] += 1
        else:
            self.seen_event_flags[bitsum] = 1
      
            
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
            self.new_pos= -.0001*self.seen_position[key]
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
        
        self.state = np.hstack([1,self.x_pos, self.y_pos, self.map_loc, self.type_of_battle, self.slot1, self.slot1_hp, self.enemy_mon, self.enemy_mon_hp, self.party_levels, self.party, self.party_hp, self.party_max_hp, self.gym_badges, self.flags]).reshape(41)
        # print(self.party_max_hp)
        # print(self.gym_badges)
        # print(self.state)
    def update_reward(self):
        
        seen_locs = len(self.seen_location)
        new_locs =  seen_locs - self.old_seen_loc


        # self.reward = self.flags+.1*len(self.seen_location)
        self.reward = -0.001*0
        if seen_locs > 1:
            self.reward += 10*new_locs
        if self.state[-1] > 0:
            self.reward += 100*self.new_flags

        self.old_seen_loc = len(self.seen_location)
        # self.reward = self.new_flags
        # self.reward = self.flags

        if self.reward > 0:
            print(self.x_pos, self.y_pos, self.reward)



    def progress_val(self):
        return self.flags+.1*len(self.seen_location)
        
    
    def save_state(self, file):
        with open(file, "wb") as f:
            self.pyboy.save_state(f)

    def load_state(self, file):
        self.pyboy.load_state(open(file,"rb"))

    def save_screenshot(self, file):
        rgba = self.pyboy.screen.ndarray
        matplotlib.image.imsave(file, rgba)

    def visualize_policy(self, file, Q, s_ind):

        rgba = self.pyboy.screen.ndarray
        plt.imshow(rgba, origin="upper")
        plt.gca().set_aspect(1)
        # plt.axis("off")

        # Go 10 spaces on either side
        # diffs = np.arange(-10, 11)
        diffs = np.arange(-5, 6)
        xyqa = np.zeros((len(diffs)**2, 4))
        x, y = self.state[[1,2]]
        act_idx = {}
        multiplier = 16
        for i, (dx, dy) in enumerate(product(diffs, diffs)):
            xp = x + dx
            yp = y + dy
            s = [xp, yp] + list(self.state[s_ind])
            Qvals = Q(torch.tensor(s, dtype=torch.float32)).detach().numpy()
            xyqa[i, 0] = xp
            xyqa[i, 1] = yp
            xyqa[i, 2] = np.max(Qvals)
            xyqa[i, 3] = int(np.argmax(Qvals))
            a = xyqa[i, 3]
            if a in act_idx:
                act_idx[a] += [i]
            else:
                act_idx[a] = [i]

        # WindowEvent.PRESS_ARROW_DOWN,
        # WindowEvent.PRESS_ARROW_LEFT,
        # WindowEvent.PRESS_ARROW_RIGHT,
        # WindowEvent.PRESS_ARROW_UP,
        # WindowEvent.PRESS_BUTTON_A,
        arrows = {0: (0, -1),
                  1: (-1, 0),
                  2: (1, 0),
                  3: (0, 1)}
        shift = 144/2
        # shift = 0
        scale = 0.5
        for a in arrows:
            if a in act_idx:
                entries = xyqa[act_idx[a], :]
                dx = entries[:, 0] - x
                dy = entries[:, 1] - y
                xpos = shift + dx*multiplier
                ypos = shift + dy*multiplier
                plt.quiver(xpos, ypos,
                        scale*arrows[a][0],scale*arrows[a][1])
        
        # press a locations
        if 4 in act_idx:
            entries = xyqa[act_idx[4], :]
            dx = entries[:, 0] - x
            dy = entries[:, 1] - y
            xpos = shift + dx*multiplier
            ypos = shift + dy*multiplier
            plt.scatter(xpos, ypos)

        plt.savefig(file)

        plt.close()

        # Scatter for q value
        dx = xyqa[:, 0] - x
        dy = xyqa[:, 1] - y
        xpos = shift + dx*multiplier
        ypos = shift + dy*multiplier
        q =  xyqa[:, 2]
        q -= q.mean()
        q /= q.std()
        plt.imshow(rgba, origin="upper")
        plt.gca().set_aspect(1)
        plt.scatter(xpos, ypos, 10*q)
        plt.savefig("q.png")
        plt.close()





    
        
        


