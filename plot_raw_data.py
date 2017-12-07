 % cd C:\Users\djcald.CSENETID\SharedCode\seizurePrediction
# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
#import mne
import os
import seaborn as sns
import copy

from metaData import environment_info as env_info
from metaData import patient_info as pat_info

#############################################################################
# %%
def conv_montage(file):
    montage_array = (''.join([chr(i) for i in file['Montage']['MontageString']])).split(' ')
    return montage_array

def get_num_chans(file):
    num_chans = int(np.sum(list(file['Montage']['Montage'])))
    return num_chans

def convert_power_data(f,num_chans):
    labels_data = list(f['power'])
    freq_names = list(f['power'][labels_data[0]])
    power_feat_names = freq_names
    power_trials = np.squeeze(([[f['power'][interest][interest_freq][0:num_chans] for interest_freq in  f['power'][interest]] for interest in labels_data]))
    return power_trials, power_feat_names

def convert_connectivity_data(file,num_chans):
    labels_data = list(f['connectivity'])
    connectivity_names = list(f['connectivity'][labels_data[0]])
    freq_names = list(f['connectivity'][labels_data[0]][connectivity_names[0]])

    corr_feat_name = list(f['connectivity'][labels_data[0]][connectivity_names[0]])
    plv_feat_name = list(f['connectivity'][labels_data[0]][connectivity_names[1]])
    psi_feat_name = list(f['connectivity'][labels_data[0]][connectivity_names[2]])

    connectivity_trials_corr = np.squeeze([[[f['connectivity'][interest]['corrs'][interest_meas][0:num_chans,0:num_chans]] for interest_meas in f['connectivity'][interest]['corrs']] for interest in labels_data])
    connectivity_trials_plv = np.squeeze([[[f['connectivity'][interest]['corrs'][interest_meas][0:num_chans,0:num_chans]] for interest_meas in f['connectivity'][interest]['plv']] for interest in labels_data])
    connectivity_trials_psi = np.squeeze([[[f['connectivity'][interest]['psi'][interest_meas][0:num_chans,0:num_chans]] for interest_meas in f['connectivity'][interest]['psi']] for interest in labels_data])

    return connectivity_trials_corr, corr_feat_name, connectivity_trials_plv, plv_feat_name, connectivity_trials_psi, psi_feat_name

def one_hot_encode(electrodes,num_chans):

    one_hot_vec = np.zeros((num_chans,1))
    one_hot_vec[electrodes] = 1
    one_hot_vec[one_hot_vec==0] = -1

    return one_hot_vec

#############################################################################

ind_int = 2

path_int = os.path.join(env_info.data_dir,pat_info.patient_names[ind_int]+pat_info.data_file_suffix)

# load in the file
f = h5py.File(path_int,'r')

# patient name
patient_name = pat_info.patient_names[ind_int]

labels_data = list(f['data'].keys())
signal_mat = np.array([f['data'][interest][:] for interest in labels_data])

plt.figure(dpi=600)
trial = 1
channel = 60
sns.set_style("white")
plt.plot(np.squeeze(signal_mat[trial][channel][50000:54000]))
plt.plot(np.squeeze(signal_mat[trial][channel+1][50000:54000]))
plt.plot(np.squeeze(signal_mat[trial][channel+3][50000:54000]))
plt.plot(np.squeeze(signal_mat[trial][channel+5][50000:54000]))
plt.plot(np.squeeze(signal_mat[trial][channel+10][50000:54000]))
plt.title('Raw ECoG Recordings for 5 different channels')
plt.xlabel('time (samples)')
plt.ylabel('voltage (microvolts)')
plt.savefig('raw_recordings')
