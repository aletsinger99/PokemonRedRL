from pyboy import PyBoy
import keyboard
# pb=open("ROM/PokemonRed.gb.state","rb")
pyboy = PyBoy('ROM/PokemonRed.gb')

pb = open("ROM/PokemonRed.gb.state","rb")
# while not pyboy.tick():
#     pass
pyboy.tick()
pyboy.load_state(pb)
while(1):
    pyboy.tick()
    
        
pyboy.stop()