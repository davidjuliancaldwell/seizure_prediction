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
# preprocessing
for ind_int in np.arange(0,num_patients):

    path_int = os.path.join(env_info.data_dir,pat_info.patient_names[ind_int]+pat_info.data_file_suffix)

    # load in the file
    f = h5py.File(path_int,'r')

    # patient name
    patient_name = pat_info.patient_names[ind_int]

    # get the number of channels
    num_chans = get_num_chans(f)

    # account for 3 not complete montages
    #if patient_name in ['fca96e','78283a', '294e1c']:
    #    num_chans = 64

    num_chans_array[ind_int] = num_chans
    num_chans_array = num_chans_array.astype(int)
    # seizure electrodes
    seizure_elec = one_hot_encode(pat_info.seizure_electrodes[ind_int],num_chans)

    # generate a random sequence for the number of channels

    num_blocks = len(f['connectivity'])

    # generate a random sequence
    num_windows_random = np.arange(num_blocks)
    random_seq_arr = np.array([np.random.permutation(num_chans) for i in num_windows_random])

    # randomly shuffle test labels
    seizure_elec_shuff = np.repeat(seizure_elec,num_blocks,axis=1).T
    seizure_elec_shuff = np.array([seizure_elec_shuff[i,random_seq_arr[i,:]] for i in np.arange(num_blocks)])
    seizure_elec_shuff = np.squeeze(np.reshape(seizure_elec_shuff,(1,-1)))

    # get the montage
    montage = conv_montage(f)

    p_t,p_n = convert_power_data(f,num_chans)
    c_corr_t,c_corr_n,c_plv_t,c_plv_n,c_psi_t,c_psi_n = convert_connectivity_data(f,num_chans)
    p_t_t = np.transpose(p_t,(0,2,1))

    c_corr_t_t = np.concatenate((np.mean(c_corr_t,axis=3),scipy.stats.skew(c_corr_t,axis=3),scipy.stats.kurtosis(c_corr_t,axis=3) ), axis = 1)
    c_psi_t_t =  np.concatenate((np.mean(c_psi_t,axis=3),scipy.stats.skew(c_psi_t,axis=3),scipy.stats.kurtosis(c_psi_t,axis=3) ), axis = 1)
    c_plv_t_t =  np.concatenate((np.mean(c_plv_t,axis=3),scipy.stats.skew(c_plv_t,axis=3),scipy.stats.kurtosis(c_plv_t,axis=3) ), axis = 1)

    c_corr_t_t = np.transpose(c_corr_t_t,(0,2,1))
    c_psi_t_t = np.transpose(c_psi_t_t,(0,2,1))
    c_plv_t_t = np.transpose(c_plv_t_t,(0,2,1))

    #c_psi_t_t = np.transpose(c_psi_t,(0,2,1,3))
    #c_psi_t_t = c_psi_t_t.reshape(c_psi_t_t.shape[0],c_psi_t_t.shape[1],-1)
    #c_corr_t_t = np.transpose(c_corr_t,(0,2,1,3))
    #_corr_t_t = c_corr_t_t.reshape(c_corr_t_t.shape[0],c_corr_t_t.shape[1],-1)
    #c_plv_t_t = np.transpose(c_plv_t,(0,2,1,3))
    #c_plv_t_t = c_plv_t_t.reshape(c_plv_t_t.shape[0],c_plv_t_t.shape[1],-1)

    data_features = np.concatenate((p_t_t,c_psi_t_t,c_corr_t_t,c_plv_t_t),axis=2)
    #data_features = p_t_t
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

# train and test split

leave = 7
total_elecs = seizure_elec_all.shape[0]
n_leave = 3*num_chans_array[leave]
index_train = np.array([ np.arange(0,np.sum(num_chans_array[0:leave])), np.arange(np.sum(num_chans_array[0:leave])+n_leave,np.sum(num_chans_array))])
train_data = shuff_data_all[0:-n_leave,:]
holdout_test_data = shuff_data_all[total_elecs-n_leave:,:]
train_labels = seizure_elec_all[0:-n_leave]
holdout_test_labels = seizure_elec_all[total_elecs-n_leave:]

#np.random.shuffle(test_labels)

# demean the data
train_data_average = np.repeat(np.array([(np.mean(train_data,axis=0))]).T,train_data.shape[0],axis=1).T
test_data_average = np.repeat(np.array([(np.mean(train_data,axis=0))]).T,test_data.shape[0],axis=1).T

train_data_std = np.repeat(np.array([(np.std(train_data,axis=0))]).T,train_data.shape[0],axis=1).T
test_data_std = np.repeat(np.array([(np.std(train_data,axis=0))]).T,test_data.shape[0],axis=1).T

# mean subtract and normalize
train_data = (train_data - train_data_average)/train_data_std

# separate the holdout data entirely, will do cross validation on the rest
holdout_test_data = (test_data - test_data_average)/test_data_std
###########################################################################
# %%
# logistic regression
with sns.axes_style("white"):
    fig1,ax1 = plt.subplots(dpi=600)
    ax1.grid(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    sns.despine(left=True,bottom=True)

    fig2,ax2 = plt.subplots(dpi=600)
    ax2.grid(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    sns.despine(left=True,bottom=True)

    fig3,ax3 = plt.subplots(dpi=600)
    ax3.grid(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    sns.despine(left=True,bottom=True)

    fig4,ax4 = plt.subplots(dpi=600)
    ax4.grid(False)
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    sns.despine(left=True,bottom=True)

sparse_vec = [100,10,1,0.1,0.01]

for i, norm_val in enumerate(sparse_vec):

    LR_mod_l1 = LogisticRegression(C=norm_val,penalty='l1',tol=0.01,class_weight="balanced")

    LR_mod_l1.fit(train_data,train_labels)
    LR_mod_l1_coeff = LR_mod_l1.coef_.ravel()
    sparsity_LR_l1 = np.mean(LR_mod_l1_coeff == 0) * 100
    print("C=%.2f" % norm_val)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity_LR_l1)
    print("train accuracy with L1 penalty: %.4f" % LR_mod_l1.score(train_data,train_labels))
    print("test accuracy with L1 penalty: %.4f" % LR_mod_l1.score(test_data,test_labels))

    train_pred = LR_mod_l1.predict(train_data)
    test_pred = LR_mod_l1.predict(test_data)
    print("train precision with L1 penalty: %.4f" % metrics.precision_score(train_labels,train_pred))
    print("test precision with L1 penalty: %.4f" % metrics.precision_score(test_labels,test_pred))

    test_score_l1 = LR_mod_l1.decision_function(test_data)
    average_precision_l1 = average_precision_score(test_labels, test_score_l1)

    print('Average precision-recall score: {0:0.2f} '.format(average_precision_l1))
    precision_l1, recall_l1, _ = precision_recall_curve(test_labels, test_score_l1)

    fpr_l1, tpr_l1, _ = roc_curve(test_labels, test_score_l1)
    roc_auc_l1 = auc(fpr_l1, tpr_l1)
    print("AUC: {0:0.2f}".format(roc_auc_l1))

    # permutation testing
    cv = StratifiedKFold(3)
    score_l1, permutation_scores_l1, pvalue_l1 = permutation_test_score(LR_mod_l1, train_data, train_labels, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)
    #print("permutation score: {0:0.2f}, p value {0:0.2f} \n".format(score_l1,pvalue_l1))
    print("permutation score: {:0.2f}, p value {:0.2f} \n".format(score_l1,pvalue_l1))

###########################################################################
    LR_mod_l2 = LogisticRegression(C=norm_val,penalty='l2',tol=0.01,class_weight="balanced")
    LR_mod_l2.fit(train_data,train_labels)
    LR_mod_l2_coeff = LR_mod_l2.coef_.ravel()
    sparsity_LR_l2 = np.mean(LR_mod_l2_coeff == 0) * 100
    print("Sparsity with L2 penalty: %.2f%%" % sparsity_LR_l2)
    print("train accuracy with L2 penalty: %.4f" % LR_mod_l2.score(train_data,train_labels))
    print("test accuracy with L2 penalty: %.4f" % LR_mod_l2.score(test_data,test_labels))

    train_pred = LR_mod_l2.predict(train_data)
    test_pred = LR_mod_l2.predict(test_data)
    print("train precision with L2 penalty: %.4f" % metrics.precision_score(train_labels,train_pred))
    print("test precision with L2 penalty: %.4f" % metrics.precision_score(test_labels,test_pred))

    test_score_l2 = LR_mod_l2.decision_function(test_data)
    average_precision_l2 = average_precision_score(test_labels, test_score_l2)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision_l2))
    precision_l2, recall_l2, _ = precision_recall_curve(test_labels, test_score_l2)

    fpr_l2, tpr_l2, _ = roc_curve(test_labels, test_score_l2)
    roc_auc_l2 = auc(fpr_l2, tpr_l2)
    print("AUC: {0:0.2f}".format(roc_auc_l2))

    l1_plot = fig2.add_subplot(5, 2, 2 * i + 1)
    l2_plot = fig2.add_subplot(5, 2, 2 * (i + 1))
    if i == 0:
        l1_plot.set_title("L1 penalty")
        l2_plot.set_title("L2 penalty")
    if i == (len(sparse_vec) - 1):
        l2_plot.set_xlabel('False Positive Rate')
        l2_plot.set_ylabel('True Positive Rate')

    lw = 2
    l1_plot.plot(fpr_l1, tpr_l1, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_l1)
    l1_plot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    l1_plot.set_ylim([0.0, 1.05])
    l1_plot.set_xlim([0.0, 1.0])
    #l1_plot.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision_l1))

    lw = 2
    l2_plot.plot(fpr_l2, tpr_l2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_l2)
    l2_plot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    l2_plot.set_ylim([0.0, 1.05])
    l2_plot.set_xlim([0.0, 1.0])

    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l2_plot.set_xticks(())
    l2_plot.set_yticks(())

    l1_plot = fig3.add_subplot(5, 2, 2 * i + 1)
    l2_plot = fig3.add_subplot(5, 2, 2 * (i + 1))
    if i == 0:
        l1_plot.set_title("L1 penalty")
        l2_plot.set_title("L2 penalty")
    if i == (len(sparse_vec) - 1):
        l2_plot.set_xlabel('Recall')
        l2_plot.set_ylabel('Precision')
    l1_plot.step(recall_l1, precision_l1, color='b', alpha=0.2,where='post')
    l1_plot.fill_between(recall_l1, precision_l1, step='post', alpha=0.2,color='b')
    l1_plot.set_ylim([0.0, 1.05])
    l1_plot.set_xlim([0.0, 1.0])
    l2_plot.step(recall_l2, precision_l2, color='b', alpha=0.2,where='post')
    l2_plot.fill_between(recall_l2, precision_l2, step='post', alpha=0.2,color='b')
    l2_plot.set_ylim([0.0, 1.05])
    l2_plot.set_xlim([0.0, 1.0])
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l2_plot.set_xticks(())
    l2_plot.set_yticks(())
    #fig2.text(-3, 2, "C = {:.2f}".format(norm_val))

    ###########################################################################
    l1_plot = fig1.add_subplot(5, 2, 2 * i + 1)
    l2_plot = fig1.add_subplot(5, 2, 2 * (i + 1))
    if i == 0:
        l1_plot.set_title("L1 penalty")
        l2_plot.set_title("L2 penalty")


    cax = l1_plot.imshow(np.abs(LR_mod_l1_coeff.reshape(1, LR_mod_l1_coeff.shape[0])), interpolation='nearest',
                   cmap='binary', vmax=1, vmin=0)
    l2_plot.imshow(np.abs(LR_mod_l2_coeff.reshape(1, LR_mod_l2_coeff.shape[0])), interpolation='nearest',
                   cmap='binary', vmax=1, vmin=0)
    #fig1.text(-3, 2, "C = {:.2f}".format(norm_val))

    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l2_plot.set_xticks(())
    l2_plot.set_yticks(())

    ##############################################################################
    # permutation testing
    score_l2, permutation_scores_l2, pvalue_l2 = permutation_test_score(LR_mod_l2, train_data, train_labels, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)
    print("permutation score: {0:0.2f}, p value {1:0.2f} \n".format(score_l2,pvalue_l2))

    # #############################################################################
    # View histogram of permutation scores
    l1_plot = fig4.add_subplot(5, 2, 2 * i + 1)
    l2_plot = fig4.add_subplot(5, 2, 2 * (i + 1))
    if i == 0:
        l1_plot.set_title("L1 penalty")
        l2_plot.set_title("L2 penalty")
    n_classes=2
    l1_plot.hist(permutation_scores_l1, 20, label='Permutation scores',
             edgecolor='black')
    ylim = l1_plot.set_ylim()
    l1_plot.vlines(score_l1, ylim[0], ylim[1], linestyle='--', color='g', linewidth=3, label='Classification Score'' (pvalue {0:0.2f})'.format(pvalue_l1))
    l1_plot.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',color='k', linewidth=3, label='Luck')
    #plt.plot(2 * [score], ylim, '--g', linewidth=3,
    #         label='Classification Score'
    #         ' (pvalue %s)' % pvalue)
    #plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

    l1_plot.set_ylim(ylim)
    l1_plot.legend()
    l1_plot.set_xlabel('Score')

    l2_plot.hist(permutation_scores_l2, 20, label='Permutation scores',
             edgecolor='black')
    ylim = l2_plot.set_ylim()
    l2_plot.vlines(score_l2, ylim[0], ylim[1], linestyle='--',color='g', linewidth=3, label='Classification Score''(pvalue {0:0.2f})'.format(pvalue_l1))
    l2_plot.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',
             color='k', linewidth=3, label='Luck')
    #plt.plot(2 * [score], ylim, '--g', linewidth=3,
    #         label='Classification Score'
    #         ' (pvalue %s)' % pvalue)
    #plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

    l2_plot.set_ylim(ylim)
    l2_plot.legend()
    l2_plot.set_xlabel('Score')

#cbar = fig1.colorbar(cax, ticks=[-1, 0, 1], orientation='vertical')

plt.gcf
#plt.savefig('milestone_different_c_comb.png')


###############################################################################
# kernel classification
# %%

# decision theory
# maximize probability of detection under constraint
# classifier with decision rule - move predictor linearly
# move that threshold until false alarm rate is ...
# tune weight of loss function until you get desired effect
# binary classification, hinge function, put big weight on that one
# search over differnet weight values until you get the false alarm rate that you want
# optimizing precision recall - pearson classifacation
#
# for neural nets with time series
# sliding window
# do a cnn on finite window

with sns.axes_style("white"):
    fig6,ax6 = plt.subplots(dpi=600)
    ax6.grid(False)
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    sns.despine(left=True,bottom=True)

    fig5,ax5 = plt.subplots(dpi=600)
    ax5.grid(False)
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    sns.despine(left=True,bottom=True)

    fig7,ax7 = plt.subplots(dpi=600)
    ax7.grid(False)
    ax7.get_xaxis().set_visible(False)
    ax7.get_yaxis().set_visible(False)
    sns.despine(left=True,bottom=True)


for ind,kernel in enumerate(('linear','poly', 'rbf')):
    sample_weight = (train_labels.shape[0]/(2*np.bincount(train_labels==1)))
    #sample_weight = np.array([0.2,9])
    keys = [0,1]
    sample_weight_dict = dict(zip(keys,sample_weight.T))
    svm_mod = svm.SVC(kernel=kernel,class_weight=sample_weight_dict)
    svm_mod.fit(train_data,train_labels)
    svm_train_score = svm_mod.score(train_data,train_labels)
    svm_test_score = svm_mod.score(test_data,test_labels)
    print('train accuracy {:.4f} for {} kernel'.format(svm_train_score,kernel))
    print('test accuracy {:.4f} for {} kernel'.format(svm_train_score,kernel))

    train_pred = svm_mod.predict(train_data)
    test_pred = svm_mod.predict(test_data)
    test_pred
    print("train precision {:.4f} for {} kernel".format(metrics.precision_score(train_labels,train_pred),kernel))
    print("test precision {:.4f} for {} kernel".format(metrics.precision_score(test_labels,test_pred),kernel))
    test_labels
    test_pred

    test_score_svm = svm_mod.decision_function(test_data)
    average_precision_svm = average_precision_score(test_labels, test_score_svm)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision_svm))
    precision_svm, recall_svm, thresh_svm = precision_recall_curve(test_labels, test_score_svm)
    svm_plot = fig5.add_subplot(3, 1, ind+1)
    fpr, tpr, _ = roc_curve(test_labels, test_score_svm)
    roc_auc_svm = auc(fpr, tpr)

    print("AUC: {0:0.2f}".format(roc_auc_svm))

    lw = 2
    svm_plot.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    svm_plot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    #svm_plot.step(recall_svm, precision_svm, color='b', alpha=0.2,where='post')
    #svm_plot.fill_between(recall_svm, precision_svm, step='post', alpha=0.2,color='b')
    svm_plot.set_ylim([0.0, 1.05])
    svm_plot.set_xlim([0.0, 1.0])
    svm_plot.set_xticks(())
    svm_plot.set_yticks(())

    svm_plot = fig6.add_subplot(3, 1, ind+1)

    lw = 2
    svm_plot.step(recall_svm, precision_svm, color='b', alpha=0.2,where='post')
    svm_plot.fill_between(recall_svm, precision_svm, step='post', alpha=0.2,color='b')
    svm_plot.set_ylim([0.0, 1.05])
    svm_plot.set_xlim([0.0, 1.0])
    svm_plot.set_xticks(())
    svm_plot.set_yticks(())

    cv = StratifiedKFold(3)
    score_svm, permutation_scores_svm, pvalue_svm = permutation_test_score(svm_mod, train_data, train_labels, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)
    print("permutation score: {0:0.2f}, p value {1:0.2f} \n".format(score_svm,pvalue_svm))

    # #############################################################################
    # View histogram of permutation scores
    svm_plot = fig7.add_subplot(3, 1, ind+1)

    n_classes=2
    svm_plot.hist(permutation_scores_svm, 20, label='Permutation scores',
             edgecolor='black')
    ylim = svm_plot.set_ylim()
    svm_plot.vlines(score_svm, ylim[0], ylim[1], linestyle='--', color='g', linewidth=3, label='Classification Score'' (pvalue {0:0.2f})'.format(pvalue_svm))
    svm_plot.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',color='k', linewidth=3, label='Luck')
    #plt.plot(2 * [score], ylim, '--g', linewidth=3,
    #         label='Classification Score'
    #         ' (pvalue %s)' % pvalue)
    #plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

    svm_plot.set_ylim(ylim)
    svm_plot.legend()
    svm_plot.set_xlabel('Score')

# gradient boosting
# %%
# Fit classifier with out-of-bag estimates

params = {'n_estimators': 1200, 'max_depth': 4, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1,'min_samples_split':4, 'random_state': 3}
clf = ensemble.GradientBoostingClassifier(**params)

sample_weight = (train_labels.shape[0]/(2*np.bincount(train_labels==1)))
#ample_weight = np.array([1,3])
sample_weight_array = np.zeros(train_labels.shape[0])
sample_weight_array[train_labels==0] = sample_weight[0]
sample_weight_array[train_labels==1] = sample_weight[1]
clf.fit(train_data,train_labels,sample_weight=sample_weight_array)

acc_train = clf.score(train_data,train_labels)
acc_test = clf.score(test_data,test_labels)

print("Train accuracy for gradient boosting: {:.4f}".format(acc_train))
print("Test accuracy for gradient boosting: {:.4f}".format(acc_test))

train_pred = clf.predict(train_data)
test_pred = clf.predict(test_data)

print("Train precision {:.4f} for gradient boosting".format(metrics.precision_score(train_labels,train_pred)))
print("Test precision {:.4f} for gradient boosting".format(metrics.precision_score(test_labels,test_pred)))

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
#plt.yticks(pos, np.array(p_n)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
#plt.savefig('gradientboost_feature_importance')

# gradient boosting optimization

param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 5, 9, 17],
              # 'max_features': [1.0, 0.3, 0.1] ## not possible in our example (only 1 fx)
              }

est = ensemble.GradientBoostingClassifier(n_estimators=3000)
# this may take some minutes
gs_cv = GridSearchCV(est, param_grid, n_jobs=4,scoring='accuracy').fit(train_data, train_labels,sample_weight=sample_weight_array)
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
#plt.yticks(pos, np.array(p_n)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


cv = StratifiedKFold(3)
score_svm, permutation_scores_svm, pvalue_svm = permutation_test_score(clf, train_data, train_labels, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)

# %%
plt.figure(dpi=600)
n_classes=2
plt.hist(permutation_scores_svm, 20, label='Permutation scores',
         edgecolor='black')
ylim = plt.ylim()
plt.vlines(score_svm, ylim[0], ylim[1], linestyle='--', color='g', linewidth=3, label='Classification Score'' (pvalue {0:0.2f})'.format(pvalue_svm))
plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',color='k', linewidth=3, label='Luck')
#plt.plot(2 * [score], ylim, '--g', linewidth=3,
#         label='Classification Score'
#         ' (pvalue %s)' % pvalue)
#plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
