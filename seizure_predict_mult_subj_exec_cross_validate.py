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
###########################################################################
# %%
# logistic regression
sns.axes_style("white")

sparse_vec = [100,50,10,5,1,0.5,0.1,0.05,0.01,0.005]
penalty_vec = ['l1','l2']
score_vec = np.zeros((len(sparse_vec),2))

#param_grid = {'C': [100,50,10,5,1,0.5,0.1,0.05,0.01,0.005], 'penalty': ['l1','l2'],'class_weight': ['balanced']}
param_grid = {'C':[100,50,10,5,1,0.5,0.1,0.05,0.01,0.005], 'penalty': ['l1','l2'] ,'class_weight': ['balanced']}

#params = ParameterGrid(param_grid)
logreg = LogisticRegression()
cv = StratifiedKFold(5)
logreg_search = GridSearchCV(logreg,param_grid = param_grid,cv=cv)
logreg_search.fit(train_data,train_labels)

best_params = logreg_search.best_params_
logreg_search.best_score_
best_log_model = LogisticRegression(**best_params)

for i, norm_val in enumerate(sparse_vec):

    LR_mod_l1 = LogisticRegression(C=norm_val,penalty='l1',tol=0.01,class_weight="balanced")
    cv = StratifiedKFold(5)
    scores_l1 = cross_validate(LR_mod_l1, train_data, train_labels, cv = cv, return_train_score=True,scoring=('accuracy','precision','average_precision','f1'))

    LR_mod_l2 = LogisticRegression(C=norm_val,penalty='l2',tol=0.01,class_weight="balanced")
    scores_l2 = cross_validate(LR_mod_l2, train_data, train_labels, cv = cv, return_train_score=True,scoring=('accuracy','precision','average_precision','f1'))

    avgDict_l1 = {}
    for k,v in scores_l1.items():
        # v is the list of grades for student k
        avgDict_l1[k] = sum(v)/ float(len(v))

    avgDict_l2 = {}
    for k,v in scores_l2.items():
        # v is the list of grades for student k
        avgDict_l2[k] = sum(v)/ float(len(v))

    score_vec[i,0] = avgDict_l1['test_accuracy']
    score_vec[i,1] = avgDict_l2['test_accuracy']

best_log_ind = np.unravel_index(score_vec.argmax(), score_vec.shape)
best_sparse = sparse_vec[best_log_ind[0]]
best_penalty = penalty_vec[best_log_ind[1]]

############################################################################
# now we have picked what's best from logistic regression, try it on all train

best_log_model = LogisticRegression(C=best_sparse,penalty=best_penalty,tol=0.01,class_weight="balanced")
best_log_model.fit(train_data,train_labels)
best_log_model_coeff = best_log_model.coef_.ravel()
sparsity_best_log_model = np.mean(best_log_model_coeff == 0) * 100

# predict probabilities
probability_log = best_log_model.predict_proba(test_data)

# what are the best features!
top10 = np.argsort(np.abs(best_log_model_coeff ))[::-1][:10]
best_features = [data_features_name[i] for i in top10]
print('Ranked Features')
for i,feat in enumerate(best_features):
    print('# {} , {} \n'.format(i,feat))

print("C=%.2f" % norm_val)
print("Sparsity: %.2f%%" % sparsity_best_log_model)
print("test accuracy: %.4f" % best_log_model.score(test_data,test_labels))

train_pred = best_log_model.predict(train_data)
test_pred = best_log_model.predict(test_data)
print("test precision: %.4f" % metrics.precision_score(test_labels,test_pred))

test_score_best_log = best_log_model.decision_function(test_data)
average_precision_best_log = average_precision_score(test_labels, test_score_best_log)
print('Average precision-recall score: {0:0.2f} '.format(average_precision_best_log))

precision_best_log, recall_best_log, _ = precision_recall_curve(test_labels, test_score_best_log)
fpr_best_log, tpr_best_log, _ = roc_curve(test_labels, test_score_best_log)
roc_auc_best_log = auc(fpr_best_log, tpr_best_log)
print("AUC: {0:0.2f}".format(roc_auc_best_log))
with sns.axes_style('darkgrid'):
    plt.figure(dpi=600)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Curve for logistic regression classifier')
    lw = 2
    plt.plot(fpr_best_log, tpr_best_log, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_best_log)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig('roc_logistic_v2')
    plt.savefig('roc_logistic_v2.svg')

plt.figure(dpi=600)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. recall graph for logistic regression classifier')
plt.step(recall_best_log, precision_best_log, color='b', alpha=0.2,where='post')
plt.fill_between(recall_best_log, precision_best_log, step='post', alpha=0.2,color='b')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.savefig('precisionrecall_logistic_v2')
plt.savefig('precisionrecall_logistic_v2.svg')


###########################################################################
plt.figure(dpi=600)
nice_size = np.append(best_log_model_coeff,np.nan)
sns.set_style('white')
plt.imshow(np.abs(nice_size.reshape(10, int(nice_size.shape[0]/10))), interpolation='nearest',cmap='binary', vmax=1, vmin=0)
plt.xticks(())
plt.yticks(())
plt.colorbar(ticks=[-1, 0, 1], orientation='vertical')
plt.title('Sparsity pattern and weights of features')
plt.savefig('sparsity_regression_v2')
plt.savefig('sparsity_regression_v2.svg')

##############################################################################
# permutation testing
score_best_log, permutation_scores_best_log, pvalue_best_log = permutation_test_score(best_log_model,train_data, train_labels, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)
print("permutation score: {0:0.2f}, p value {1:0.2f} \n".format(score_best_log,pvalue_best_log))

# #############################################################################
# View histogram of permutation scores
sns.set_style('dark')
plt.figure(dpi=600)
n_classes=2
#plt.hist(permutation_scores_best_log, 20, label='Permutation scores',
#         edgecolor='black')
sns.distplot(permutation_scores_best_log, 20, label='Permutation scores',kde=False)
ylim = plt.ylim()
plt.vlines(score_best_log, ylim[0], ylim[1], linestyle='--', color='g', linewidth=3, label='Classification Score'' \n (pvalue {0:0.2f})'.format(pvalue_best_log))
plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',color='k', linewidth=3, label='Luck')
plt.ylim(ylim)
plt.ylabel('Count')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(loc='upper left')
plt.xlabel('Score')
plt.xlim((0,1))
plt.title('Permutation testing of logistic regression classifier')
plt.savefig('permutation_testing_logistic_v2')
plt.savefig('permutation_testing_logistic_v2.svg')

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
kernel_vector = ['linear','poly','rbf','sigmoid']
score_vec_svm = np.zeros((len(kernel_vector),1))
for ind,kernel in enumerate(kernel_vector):

    sample_weight = (train_labels.shape[0]/(2*np.bincount(train_labels==1)))
    #sample_weight = np.array([0.2,9])
    keys = [0,1]
    sample_weight_dict = dict(zip(keys,sample_weight.T))
    svm_mod = svm.SVC(kernel=kernel,class_weight=sample_weight_dict)
    cv = StratifiedKFold(5)
    scores_svm = cross_validate(svm_mod, train_data, train_labels, cv = cv, return_train_score=True,scoring=('accuracy','precision','average_precision','f1'))

    avgDict_svm = {}
    for k,v in scores_svm.items():
        # v is the list of grades for student k
        avgDict_svm[k] = sum(v)/ float(len(v))

    score_vec_svm[ind] = avgDict_svm['test_accuracy']

# %%
best_svm_ind = np.unravel_index(score_vec_svm.argmax(), score_vec_svm.shape)
best_kernel = kernel_vector[best_svm_ind[0]]


sample_weight = (train_labels.shape[0]/(2*np.bincount(train_labels==1)))
#sample_weight = np.array([0.2,9])
keys = [0,1]
sample_weight_dict = dict(zip(keys,sample_weight.T))
svm_mod = svm.SVC(kernel=best_kernel,class_weight=sample_weight_dict,probability=True)
svm_mod.fit(train_data,train_labels)

support = svm_mod.support_
support_vectors = svm_mod.support_vectors_
num_support = svm_mod.n_support_

probability_svm = svm_mod.predict_proba(test_data)
svm_train_score = svm_mod.score(train_data,train_labels)
svm_test_score = svm_mod.score(test_data,test_labels)
print('test accuracy: {:.4f} for {} kernel'.format(svm_train_score,kernel))

train_pred = svm_mod.predict(train_data)
test_pred = svm_mod.predict(test_data)
print("test precision: {:.4f} for {} kernel".format(metrics.precision_score(test_labels,test_pred),kernel))

test_score_svm = svm_mod.decision_function(test_data)
average_precision_svm = average_precision_score(test_labels, test_score_svm)

print('Average precision-recall score: {0:0.2f}'.format(average_precision_svm))
precision_svm, recall_svm, thresh_svm = precision_recall_curve(test_labels, test_score_svm)
fpr_svm, tpr_svm, _ = roc_curve(test_labels, test_score_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

print("AUC: {0:0.2f}".format(roc_auc_svm))

with sns.axes_style('darkgrid'):
    plt.figure(dpi=600)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Curve for SVM classifier')
    lw = 2
    plt.plot(fpr_svm, tpr_svm, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_svm)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig('roc_svm')
    plt.savefig('roc_svm.svg')

plt.figure(dpi=600)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. recall graph for SVM classifier')
plt.step(recall_svm, precision_svm, color='b', alpha=0.2,where='post')
plt.fill_between(recall_svm, precision_svm, step='post', alpha=0.2,color='b')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.savefig('precisionrecall_svm')
plt.savefig('precisionrecall_svm.svg')

###########################################################################

# %%
##############################################################################
# permutation testing
score_best_svm, permutation_scores_best_svm, pvalue_best_svm = permutation_test_score(svm_mod,train_data, train_labels, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)
print("permutation score: {0:0.2f}, p value {1:0.2f} \n".format(score_best_svm,pvalue_best_svm))
# %%
# #############################################################################
# View histogram of permutation scores
with sns.axes_style('dark'):
    plt.figure(dpi=600)
    n_classes=2
    #plt.hist(permutation_scores_best_log, 20, label='Permutation scores',
    #         edgecolor='black')
    sns.distplot(permutation_scores_best_svm, 20, label='Permutation scores',kde=False)
    ylim = plt.ylim()
    plt.vlines(score_best_svm, ylim[0], ylim[1], linestyle='--', color='g', linewidth=3, label='Classification Score'' \n (pvalue {0:0.2f})'.format(pvalue_best_svm))
    plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',color='k', linewidth=3, label='Luck')
    plt.ylim(ylim)
    plt.ylabel('Count')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(loc='upper left')
    plt.xlabel('Score')
    plt.xlim((0,1))
    plt.title('Permutation testing of SVM classifier')
    plt.savefig('permutation_testing_svm')
    plt.savefig('permutation_testing_svm.svg')

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
# %%
# gradient boosting optimization

param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 5, 9, 17],
              # 'max_features': [1.0, 0.3, 0.1] ## not possible in our example (only 1 fx)
              }

est = ensemble.GradientBoostingClassifier(n_estimators=3000)
# this may take some minutes
gs_cv = GridSearchCV(est, param_grid, n_jobs=4,scoring='precision').fit(train_data, train_labels,sample_weight=sample_weight_array)
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
# %%
# Plot feature importance
plt.figure(dpi=600)
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(data_features_name)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('gb_feat_import')
plt.show()
# %%

cv = StratifiedKFold(5)
score_gb, permutation_scores_gb, pvalue_gb = permutation_test_score(clf, train_data, train_labels, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)
# %%
plt.figure(dpi=600)
n_classes=2
plt.hist(permutation_scores_gb, 20, label='Permutation scores',
         edgecolor='black')
ylim = plt.ylim()
plt.vlines(score_gb, ylim[0], ylim[1], linestyle='--', color='g', linewidth=3, label='Classification Score'' (pvalue {0:0.2f})'.format(pvalue_gb))
plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',color='k', linewidth=3, label='Luck')
#plt.plot(2 * [score], ylim, '--g', linewidth=3,
#         label='Classification Score'
#         ' (pvalue %s)' % pvalue)
#plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
