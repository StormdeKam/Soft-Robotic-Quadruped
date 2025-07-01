import numpy as np
import time

t0 = time.time()
arr = np.load("rl/datasets/no_steady_states/demo_states_merged.npy", mmap_mode='r')
print("Load time:", time.time() - t0)
print("Shape:", arr.shape)

t1 = time.time()
_ = arr[:].copy()  # actually touch the memory
print("Full array read time:", time.time() - t1)
