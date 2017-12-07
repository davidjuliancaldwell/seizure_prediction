# %%
#############################################################################
# setup working environement
% cd C:\Users\djcald.CSENETID\SharedCode\seizurePrediction
#% cd C:\Users\David\Research\seizure_prediction

% matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import h5py
#import mne
import os
import seaborn as sns
import copy
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn import ensemble
from sklearn import utils
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score


from metaData import environment_info as env_info
from metaData import patient_info as pat_info


sns.set()
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.25
#############################################################################
# define functions of interest
#%%

def conv_montage(file):
    montage_array = (''.join([chr(i) for i in file['Montage']['MontageString']])).split(' ')
    return montage_array

def get_num_chans(file):
    #num_chans = int(np.sum(list(file['Montage']['Montage'])))
    num_chans = int(f['data']['solo'].shape[0])
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
    if len(electrodes)>0:
        one_hot_vec[electrodes] = 1
        one_hot_vec[one_hot_vec==0] = 0

    return one_hot_vec
#############################################################################

# iterate over all patients
# %%

num_patients = len(pat_info.patient_names)
num_chans_array = np.zeros(num_patients)
seizure_elec_array = np.zeros(num_patients)
random_seq_vec = []
# preprocessing
for ind_int in np.arange(0,num_patients):

    path_int = os.path.join(env_info.data_dir,pat_info.patient_names[ind_int]+pat_info.data_file_suffix)

    # load in the file
    f = h5py.File(path_int,'r')

    # patient name
    patient_name = pat_info.patient_names[ind_int]

    # get the number of channels
    num_chans = get_num_chans(f)
    num_chans_array[ind_int] = num_chans

    # seizure electrodes
    seizure_elec = one_hot_encode(pat_info.seizure_electrodes[ind_int],num_chans)
    seizure_elec_array[ind_int] = np.sum(seizure_elec)
    # generate a random sequence for the number of channels
    num_blocks = len(f['connectivity'])

    # generate a random sequence
    num_windows_random = np.arange(num_blocks)
    random_seq_arr = np.array([np.random.permutation(num_chans) for i in num_windows_random])
    random_seq_vec.append(random_seq_arr)
    # randomly shuffle test labels
    seizure_elec_shuff = np.repeat(seizure_elec,num_blocks,axis=1).T
    seizure_elec_shuff = np.array([seizure_elec_shuff[i,random_seq_arr[i,:]] for i in np.arange(num_blocks)])
    seizure_elec_shuff = np.squeeze(np.reshape(seizure_elec_shuff,(1,-1)))

    # get the montage
    montage = conv_montage(f)

    # conver the data
    p_t,p_n = convert_power_data(f,num_chans)
    c_corr_t,c_corr_n,c_plv_t,c_plv_n,c_psi_t,c_psi_n = convert_connectivity_data(f,num_chans)
    p_t_t = np.transpose(p_t,(0,2,1))

    c_corr_t_t = np.concatenate((np.mean(c_corr_t,axis=3),scipy.stats.skew(c_corr_t,axis=3),scipy.stats.kurtosis(c_corr_t,axis=3) ), axis = 1)
    c_psi_t_t =  np.concatenate((np.mean(c_psi_t,axis=3),scipy.stats.skew(c_psi_t,axis=3),scipy.stats.kurtosis(c_psi_t,axis=3) ), axis = 1)
    c_plv_t_t =  np.concatenate((np.mean(c_plv_t,axis=3),scipy.stats.skew(c_plv_t,axis=3),scipy.stats.kurtosis(c_plv_t,axis=3) ), axis = 1)

    c_corr_t_t = np.transpose(c_corr_t_t,(0,2,1))
    c_psi_t_t = np.transpose(c_psi_t_t,(0,2,1))
    c_plv_t_t = np.transpose(c_plv_t_t,(0,2,1))

    # concatenate data features
    data_features = np.concatenate((p_t_t,c_psi_t_t,c_corr_t_t,c_plv_t_t),axis=2)

    # shuffle the data
    random_seq_data = np.reshape(np.repeat(random_seq_arr,data_features.shape[2],axis=1),np.array(data_features.shape))
    shuff_data = np.zeros((data_features.shape))

    for i in np.arange(data_features.shape[0]):
        shuff_data[i,:] = data_features[i,random_seq_arr[i]]

    if ind_int == 0:
        shuff_data_all = shuff_data.reshape((shuff_data.shape[0]*shuff_data.shape[1],shuff_data.shape[2]))
        seizure_elec_all = seizure_elec_shuff

    shuff_data = shuff_data.reshape((shuff_data.shape[0]*shuff_data.shape[1],shuff_data.shape[2]))

    shuff_data_all = np.vstack((shuff_data_all,shuff_data))
    seizure_elec_all = np.hstack((seizure_elec_all,seizure_elec_shuff))

# generator expression to make list of feature names
add_feat = ['_mean','_skew','_kurtosis']
c_corr_n_list = [x + '_corr' + add_feat[0] for x in c_corr_n] + [x + '_corr' + add_feat[1] for x in c_corr_n] + [x + '_corr' +add_feat[2] for x in c_corr_n]
c_psi_n_list = [x + '_psi' + add_feat[0] for x in c_psi_n] + [x +'_psi' + add_feat[1] for x in c_psi_n] + [x +'_psi'+ add_feat[2] for x in c_psi_n]
c_plv_n_list = [x +'_plv' + add_feat[0] for x in c_plv_n] + [x +'_plv' +  add_feat[1] for x in c_plv_n] + [x +'_plv'+ add_feat[2] for x in c_plv_n]

data_features_name = p_n + c_psi_n_list + c_corr_n_list + c_plv_n_list

# convert arrays of the number of channels in each array, as well as the
# seizure electrodes into arrays of integers
num_chans_array = num_chans_array.astype(int)
seizure_elec_array = seizure_elec_array.astype(int)

shuff_data_all_avg = np.repeat(np.array([(np.mean(shuff_data_all,axis=0))]).T,shuff_data_all.shape[0],axis=1).T
shuff_data_all_std = np.repeat(np.array([(np.std(shuff_data_all,axis=0))]).T,shuff_data_all.shape[0],axis=1).T
shuff_data_all = (shuff_data_all - shuff_data_all_avg)/shuff_data_all_std

cumulative_sum = np.cumsum(3*num_chans_array)
# train and test split
leave = 7
total_elecs = seizure_elec_all.shape[0]
n_leave = 3*num_chans_array[leave]
index_train = np.array([ np.arange(0,3*np.sum(num_chans_array[0:leave])), np.arange(3*np.sum(num_chans_array[0:leave])+n_leave,(3*np.sum(num_chans_array)-10))])
train_data = shuff_data_all[0:-n_leave,:]
test_data = shuff_data_all[total_elecs-n_leave:,:]
train_labels = seizure_elec_all[0:-n_leave]
test_labels = seizure_elec_all[total_elecs-n_leave:]

# demean the data
#train_data_average = np.repeat(np.array([(np.mean(train_data,axis=0))]).T,train_data.shape[0],axis=1).T
#test_data_average = np.repeat(np.array([(np.mean(train_data,axis=0))]).T,test_data.shape[0],axis=1).T

#train_data_std = np.repeat(np.array([(np.std(train_data,axis=0))]).T,train_data.shape[0],axis=1).T
#test_data_std = np.repeat(np.array([(np.std(train_data,axis=0))]).T,test_data.shape[0],axis=1).T

# mean subtract and normalize
#train_data = (train_data - train_data_average)/train_data_std

# separate the holdout data entirely, will do cross validation on the rest
#test_data = (test_data - test_data_average)/test_data_std

# gradient boosting
# %%
# Fit classifier with out-of-bag estimates


sample_weight = (train_labels.shape[0]/(2*np.bincount(train_labels==1)))
sample_weight_array = np.zeros(train_labels.shape[0])
sample_weight_array[train_labels==0] = sample_weight[0]
sample_weight_array[train_labels==1] = sample_weight[1]

param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 5, 9, 17],
              # 'max_features': [1.0, 0.3, 0.1] ## not possible in our example (only 1 fx)
              }
cv = StratifiedKFold(5)

est = ensemble.GradientBoostingClassifier(n_estimators=3000)
# this may take some minutes
gs_cv = GridSearchCV(est, param_grid,cv =cv, n_jobs=4,scoring='precision').fit(train_data, train_labels,sample_weight=sample_weight_array)
# best hyperparameter setting
gs_cv.best_params_

best_params = gs_cv.best_params_

clf = ensemble.GradientBoostingClassifier(**best_params)
clf.fit(train_data,train_labels,sample_weight=sample_weight_array)

acc_train = clf.score(train_data,train_labels)
acc_test = clf.score(test_data,test_labels)

print("Train accuracy for gradient boosting: {:.4f}".format(acc_train))
print("Test accuracy for gradient boosting: {:.4f}".format(acc_test))

train_pred = clf.predict(train_data)
test_pred = clf.predict(test_data)

print("Train precision {:.4f} for gradient boosting".format(metrics.precision_score(train_labels,train_pred)))
print("Test precision {:.4f} for gradient boosting".format(metrics.precision_score(test_labels,test_pred)))

test_score_gb = clf.decision_function(test_data)
average_precision_gb = average_precision_score(test_labels, test_score_gb)
# %%
sample_weight_test = (test_labels.shape[0]/(2*np.bincount(test_labels==1)))
sample_weight_array_test = np.zeros(test_labels.shape[0])
sample_weight_array_test[test_labels==0] = sample_weight_test[0]
sample_weight_array_test[test_labels==1] = sample_weight_test[1]

print('Average precision-recall score: {0:0.2f}'.format(average_precision_gb))
precision_gb, recall_gb, thresh_gb = precision_recall_curve(test_labels, test_score_gb)
fpr_gb, tpr_gb, _ = roc_curve(test_labels, test_score_gb,sample_weight=sample_weight_array_test)
roc_auc_gb =  roc_auc_score(test_labels,test_score_gb,sample_weight=sample_weight_array_test)

print("AUC: {0:0.2f}".format(roc_auc_gb))
# %%
with sns.axes_style('darkgrid'):
    plt.figure(dpi=600)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Curve for Gradient Boosting classifier')
    lw = 2
    plt.plot(fpr_gb, tpr_gb, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_gb)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig('roc_gb')
    plt.savefig('roc_gb.svg')
# %%
# plot feature importance
# Plot feature importance
plt.figure(dpi=600)
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
#plt.yticks(pos, np.array(data_features_name)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('gb_feat_import')
plt.show()
# %%
# what are the best features!
top10 = np.argsort(np.abs(feature_importance ))[::-1][:10]
best_features = [data_features_name[i] for i in top10]
print('Ranked Features')
for i,feat in enumerate(best_features):
    print('# {} , {} \n'.format(i,feat))

cv = StratifiedKFold(5)
score_gb, permutation_scores_gb, pvalue_gb = permutation_test_score(clf, train_data, train_labels, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)
# %%
with sns.axes_style('dark'):
    plt.figure(dpi=600)
    n_classes=2
    #plt.hist(permutation_scores_best_log, 20, label='Permutation scores',
    #         edgecolor='black')
    sns.distplot(permutation_scores_gb, 20, label='Permutation scores',kde=False)
    ylim = plt.ylim()
    plt.vlines(score_gb, ylim[0], ylim[1], linestyle='--', color='g', linewidth=3, label='Classification Score'' \n (pvalue {0:0.2f})'.format(pvalue_gb))
    plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',color='k', linewidth=3, label='Luck')
    plt.ylim(ylim)
    plt.ylabel('Count')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(loc='upper left')
    plt.xlabel('Score')
    plt.xlim((0,1))
    plt.title('Permutation testing of Gradient Boosting classifier')
    plt.savefig('permutation_testing_gb')
    plt.savefig('permutation_testing_gb.svg')
