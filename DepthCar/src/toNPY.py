import numpy as np
from Create_Data_Liet import create_data
angledata = []
with open('../data/data.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        angledata.append(int(line))
angle = np.array(angledata)
np.save('../data/data.npy', angle, False)
create_data()
