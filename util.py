import numpy as np

def round_power_2(number):
    if number <= 0:
        return 0
    power = np.floor(np.log2(number)) - 1
    return int(np.min([2 ** power, 65536]))