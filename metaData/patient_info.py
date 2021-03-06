import numpy as np
patient_names = ['0b5a2e','69da36','294e1c','78283a','a07793','c5a5e9','ecb43e','fca96e']
# 294e1c
# G 33, 41, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 62, 63, 64
# LAT 1, 2, 4
# LPT 3, 4, 5
# LTO 5, 6
# LMT 2

# done subs - 69da36, a07793, c5a5e9, maybe ecb43e
# just use grid electrodes for fca96e,78283a, 294e1c

seizure_electrodes = [
[81,82,83,84,85,87,90,95],
[7,8,15,16,29,30,34,35,44,45,54,55,63,64,79],
[33, 41, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 62, 63, 64,73,74,76,80,85,86,87], # exclude 93,94 since they aren't in data
[8,16,24,65,66,74],
[69,72,73,74,75,86,87,98,99,100],
[17,18,25,26,27,28,29,35,36,37,74,82],
[20,21,28,29,36], # exclude 84
[13, 21, 43, 44, 50, 53] # exclude 93,94,99,100,105
]

seizure_electrodes = [np.array(seizure_electrodes[i])-1 for i in np.arange(len(seizure_electrodes))]
data_file_suffix  = '_powerAndConnectivity.mat'
#
# Ecb43e
# G20, 21, 28, 29, 36
# LAD4
#
# 69da36
# LTP1
# G 7, 8, 15, 16, 29, 30, 34, 35, 44, 45, 54, 55, 63, 64
#
# C5a5e9
# RAD4, RPD4
# G 17, 18, 25, 26, 27, 28, 29, 35, 36, 37
#
# fca96e
# RTP 1,2,7
# RMT 1, 2
# G 13, 21, 43, 44, 50, 53
#
# 294e1c
# G 33, 41, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 62, 63, 64
# LAT 1, 2, 4
# LPT 3, 4, 5
# LTO 5, 6
# LMT 2
#
# 0b5a2e
# LAT 1, 2, 3, 4, 5
# LMT 1, 4
# LPT3
#
# a07793
# LAT 2, 3
# LPT 2, 3, 4
# LAF 6, 10, 11, 12, 13
#
# 78283a
# LAT1, 2
# LMT4
# G8, 16, 24
#
# 3f2113
# G 26, 29, 34, 35, 38
