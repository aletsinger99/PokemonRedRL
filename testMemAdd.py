def read_m(addr):
        return pyboy.get_memory_value(addr)
import memoryAddresses
from pyboy import PyBoy
import math
pyboy = PyBoy('ROM/PokemonRed.gb')
# while not pyboy.tick():
#     pass

while(1):
    pyboy.tick()
    mysum = 0
    for i in range(55111,55431):
        mysum = mysum + ((read_m(i)).bit_count())
        # print(hex(i))

    print('X: '+str(read_m(memoryAddresses.X_POS_ADDRESS))+'  Y:  '+str(read_m(memoryAddresses.Y_POS_ADDRESS))+ '   Map Loc:  '+ str(read_m(memoryAddresses.MAP_N_ADDRESS)) + '  Type of battle: '+str(read_m(memoryAddresses.CAN_CATCH))+ ' Mon slot 1: '+str(read_m(memoryAddresses.POKEMON_1))+ ' Mon Health: ' + 
    str(int(str(bin(read_m(memoryAddresses.POKEMON_1_H1)))[2:]+str(bin(read_m(memoryAddresses.POKEMON_1_H2)))[2:],2))+ ' Enemy Mon: '+str(read_m(memoryAddresses.E_POKEMON))+ ' EMon Health: ' + str(int(str(bin(read_m(memoryAddresses.E_POKEMON_H1)))[2:]+str(bin(read_m(memoryAddresses.E_POKEMON_H2)))[2:],2))+' flag summation: '+str(mysum))
    pyboy.stop()
    # mysum = read_m(55112)
    # print('flag summation: '+str(mysum))