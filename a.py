import csv
import numpy as np
Init_Path = './2023-05-28'
Folder_Path=f'{Init_Path}/GA/0'
convergence=np.loadtxt(f'{Folder_Path}/convergence.csv')
print(convergence)