from ac_solver.envs.ac_moves import *
import numpy as np

# example usage
pres = np.array([1, -2, 2, 0, 0, 0, -1, 2, -2, 0, 0, 0])
# (xYy, XyY)
max_len = 6
lengths = [3, 3]
action = (1, 0, 0, False)  # concatenate r_1 with r_0
new_pres, new_lengths = ACMove(action, pres, max_len, lengths)
print("Original presentation:", pres)
print("New presentation:", new_pres)
print("Original lengths:", lengths)
print("New lengths:", new_lengths)