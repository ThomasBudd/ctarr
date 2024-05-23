import numpy as np
from scipy.ndimage.filters import gaussian_filter

'''
This script contains hardcoded parameters that are needed to run the inference
'''

# preprocessing parameters
CT_WINDOWS = [[-500, 1300], [-1024, 150], [-150, 250]]
CT_SCALINGS = [[-500, 1800], [-1024, 1174], [-150, 400]]
TARGET_SPACING = np.array([3,3,3])

# sliding window parameters
PATCH_SIZE = np.array([64, 64, 64])
PATCH_OVERLAP = 0.5
tmp = np.zeros(PATCH_SIZE)
center_coords = [i // 2 for i in PATCH_SIZE]
sigmas = [i / 8  for i in PATCH_SIZE]
tmp[tuple(center_coords)] = 1
PATCH_WEIGHT = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
PATCH_WEIGHT = PATCH_WEIGHT / np.max(PATCH_WEIGHT) * 1
PATCH_WEIGHT = PATCH_WEIGHT[None].astype(np.float32)
PATCH_WEIGHT[PATCH_WEIGHT == 0] = np.min(PATCH_WEIGHT[PATCH_WEIGHT != 0])

# parameters of iterative matching
LR0 = 0.05
LR_FAC = 0.5
N_LR_RED = 4
DEL_LOSS = 0.005

