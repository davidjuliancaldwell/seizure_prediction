import numpy as np
import matplotlib.pyplot as plt
import h5py
#import mne
import os
import seaborn as sns
import copy
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV

from environment_info import *
from patient_info import *
#import environment_info

# now we have data_dir, scripts_dir, and root_dir

# functions to convert load and process the data
sns.set()
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.25

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
# iterate over the subjects

# define path of interest for a particular patient
for ind_int in np.arange(1,len(patient_names)):

    path_int = os.path.join(data_dir,patient_names[ind_int]+data_file_suffix)

    # load in the file
    f = h5py.File(path_int,'r')

    # patient name
    patient_name = patient_names[ind_int]


    # get the number of channels
    num_chans = get_num_chans(f)

    # account for 3 not complete montages
    if patient_name in ['fca96e','78283a', '294e1c']:
        num_chans = 64

    # seizure electrodes
    seizure_elec = one_hot_encode(seizure_electrodes[ind_int],num_chans)

    # generate a random sequence for the number of channels

    num_blocks = len(f['connectivity'])

    # generate a random sequence
    num_windows_random = np.arange(num_blocks)
    random_seq_arr = np.array([np.random.permutation(num_chans) for i in num_windows_random])

    # randomly shuffle test labels
    seizure_elec_shuff = np.repeat(seizure_elec,num_blocks,axis=1).T
    seizure_elec_shuff = np.array([seizure_elec_shuff[i,random_seq_arr[i,:]] for i in np.arange(num_blocks)])

    # get the montage
    montage = conv_montage(f)

    p_t,p_n = convert_power_data(f,num_chans)
    c_corr_t,c_corr_n,c_plv_t,c_plv_n,c_psi_t,c_psi_n = convert_connectivity_data(f,num_chans)

    p_t_t = np.transpose(p_t,(0,2,1))


    random_seq_data = np.reshape(np.repeat(random_seq_arr,p_t_t.shape[2],axis=1),np.array(p_t_t.shape))

    shuff_data = zeros((p_t_t.shape))

    for i in np.arange(p_t_t.shape[0]):
        shuff_data[i,:] = p_t_t[i,random_seq_arr[i]]

    train_data = shuff_data[0:2]
    test_data = shuff_data[2]
    test_data.shape

    train_labels = seizure_elec_shuff[0:2]
    test_labels = seizure_elec_shuff[2]

    if len(test_data.shape)<3:
        test_data = np.expand_dims(test_data,axis=0)


    train_data_stack = np.reshape(train_data,(train_data.shape[0]*train_data.shape[1],train_data.shape[2]))
    train_labels = np.squeeze(np.reshape(train_labels,(1,-1)))
    test_data_stack = np.reshape(test_data,(test_data.shape[0]*test_data.shape[1],test_data.shape[2]))
    test_labels = np.squeeze(np.reshape(test_labels,(1,-1)))

    # demean the data
    train_data_stack_average = np.repeat(np.array([(np.mean(train_data_stack,axis=0))]).T,train_data_stack.shape[0],axis=1).T
    test_data_stack_average = np.repeat(np.array([(np.mean(train_data_stack,axis=0))]).T,test_data_stack.shape[0],axis=1).T

    train_data_stack_std = np.repeat(np.array([(np.std(train_data_stack,axis=0))]).T,train_data_stack.shape[0],axis=1).T
    test_data_stack_std = np.repeat(np.array([(np.std(train_data_stack,axis=0))]).T,test_data_stack.shape[0],axis=1).T

    # mean subtract and normalize
    train_data= (train_data_stack - train_data_stack_average)/train_data_stack_std
    test_data= (test_data_stack - test_data_stack_average)/test_data_stack_std

    print('Patient number is {}'.format(patient_name))

    figs = {}
    for i, norm_val in enumerate((100, 1, 0.01)):

        LR_mod_l1 = LogisticRegression(C=norm_val,penalty='l1',tol=0.01)
        LR_mod_l1.fit(train_data,train_labels)
        LR_mod_l1_coeff = LR_mod_l1.coef_.ravel()
        sparsity_LR_l1 = np.mean(LR_mod_l1_coeff == 0) * 100
        print("C=%.2f" % norm_val)
        print("Sparsity with L1 penalty: %.2f%%" % sparsity_LR_l1)
        print("train score with L1 penalty: %.4f" % LR_mod_l1.score(train_data,train_labels))
        print("test score with L1 penalty: %.4f" % LR_mod_l1.score(test_data,test_labels))


        LR_mod_l2 = LogisticRegression(C=norm_val,penalty='l2',tol=0.01)
        LR_mod_l2.fit(train_data,train_labels)
        LR_mod_l2_coeff = LR_mod_l2.coef_.ravel()
        sparsity_LR_l2 = np.mean(LR_mod_l2_coeff == 0) * 100
        print("Sparsity with L2 penalty: %.2f%%" % sparsity_LR_l2)
        print("train score with L2 penalty: %.4f" % LR_mod_l2.score(train_data,train_labels))
        print("test score with L2 penalty: %.4f \n" % LR_mod_l1.score(test_data,test_labels))

        figs[i] = plt.figure(dpi=600)
        l1_plot = plt.subplot(3, 2, 2 * i + 1)
        l2_plot = plt.subplot(3, 2, 2 * (i + 1))
        if i == 0:
            l1_plot.set_title("L1 penalty")
            l2_plot.set_title("L2 penalty")

        l1_plot.imshow(np.abs(LR_mod_l1_coeff.reshape(1, 12)), interpolation='nearest',
                       cmap='binary', vmax=1, vmin=0)
        l2_plot.imshow(np.abs(LR_mod_l2_coeff.reshape(1, 12)), interpolation='nearest',
                       cmap='binary', vmax=1, vmin=0)
        plt.text(-3, 3, "C = {:.2f}".format(norm_val))

        l1_plot.set_xticks(())
        l1_plot.set_yticks(())
        l2_plot.set_xticks(())
        l2_plot.set_yticks(())

    plt.savefig('milestone_different_c_ind.png')


    svm_mod = svm.SVC(kernel='rbf')
    svm_mod.fit(train_data,train_labels)
    svm_train_score = svm_mod.score(train_data,train_labels)
    svm_test_score = svm_mod.score(test_data,test_labels)
    print('train score {} for {} kernel'.format(svm_train_score,svm_mod.kernel))
    print('test score {} for {} kernel'.format(svm_train_score,svm_mod.kernel))

    svm_mod = svm.SVC(kernel='linear')
    svm_mod.fit(train_data,train_labels)
    svm_train_score = svm_mod.score(train_data,train_labels)
    svm_test_score = svm_mod.score(test_data,test_labels)
    print('train score {} for {} kernel'.format(svm_train_score,svm_mod.kernel))
    print('test score {} for {} kernel'.format(svm_train_score,svm_mod.kernel))

    svm_mod = svm.SVC(kernel='poly')
    svm_mod.fit(train_data,train_labels)
    svm_train_score = svm_mod.score(train_data,train_labels)
    svm_test_score = svm_mod.score(test_data,test_labels)
    print('train score {} for {} kernel'.format(svm_train_score,svm_mod.kernel))
    print('test score {} for {} kernel \n'.format(svm_train_score,svm_mod.kernel))
