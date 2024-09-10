import numpy as np

x1_ini, x2_ini, x3_ini, x4_ini = np.random.rand(4, 1) 
print(np.expand_dims(np.concatenate((x1_ini, x2_ini, x3_ini, x4_ini), axis=0), axis=0))