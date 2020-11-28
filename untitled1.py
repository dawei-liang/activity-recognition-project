# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:34:31 2020

@author: david
"""


import numpy as np
import pandas as pd
import os, shutil, re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from numpy.random import seed
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeavePGroupsOut
import joblib

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.random import set_seed
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

import check_dirs


#%%
    
def last_4chars(x):    
    return(int(x.split('/')[-1].split('-')[0][4:]))
    
def last_4chars2(x):    
    return(int(x.split('/')[-1].split('_')[0][4:]))
    
#%%


def load_csv(csv_list):
    """
    Obtain feat and labels from list of csv spectrogram
    
    args:
        list of csv paths
    return:
        data and labels in two lists 
    """
    feat, labels = [], []
    for csv_item in sorted(csv_list):
        spectrogram = pd.read_csv(csv_item, header=None).values
        # naming format of the csv: /../activity\label_seg.csv
        label = csv_item.split('/')[-1].split('_')[0][-1]
        print('loaded spectrogram shape:', spectrogram.shape, 'label:', label)
        feat.append(spectrogram)
        labels.append(label)
    return feat, labels
#%%
    
#class model_nn():
#    def __init__(self, input_shape):
#        self.input_shape = input_shape
#    def model(self):    
#        model = Sequential()
#        model.add(L.Conv1D(64, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
#        model.add(L.Conv1D(128, strides=2, kernel_size=4, activation='relu', padding='same'))
#        model.add(L.Conv1D(256, strides=2, kernel_size=4, activation='relu', padding='same'))
#        model.add(L.Conv1D(512, strides=2, kernel_size=4, activation='relu', padding='same'))
#        model.add(L.Flatten())
#        model.add(L.Dense(1024, activation='relu'))
#        model.add(L.Dropout(0.2))
#        model.add(L.Dense(128, activation='relu'))
#        model.add(L.Dropout(0.2))
#        model.add(L.Dense(1, activation='sigmoid'))
#    
#        # Compile the model
#        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#        model.compile(loss=binary_crossentropy,
#                      optimizer=opt,
#                      metrics=['binary_accuracy'])
#        return model
    
class model_nn():
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def model(self):    
        model = Sequential()
        model.add(L.Conv2D(64, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(L.Conv2D(128, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Conv2D(256, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Conv2D(512, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Flatten())
        model.add(L.Dense(1024, activation='relu'))
        model.add(L.Dropout(0.2))
        model.add(L.Dense(128, activation='relu'))
        model.add(L.Dropout(0.2))
        model.add(L.Dense(1, activation='sigmoid'))
    
        # Compile the model
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=binary_crossentropy,
                      optimizer=opt,
                      metrics=['binary_accuracy'])
        return model
    
#%%
sound_dir = './'
sub_list = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
feat_size=1000
sound_list = []
# obtain the path to all csv files
for sub in sub_list:
    sound_dir = './%s/embedding_1s/' % sub
    sound_list_sub = [os.path.join(sound_dir, item) \
                      for item in os.listdir(sound_dir) if item.endswith('.csv')]
    sound_list.extend(sound_list_sub)
        
sound_data, groups = [], []   # groups is the subject index for each instance
for item in sound_list:
    sub_idx = int(item.split('/')[-3][1:])   # item: .../P1/embedding_1s/outdoor_refine.csv, extract x from Px
    audio_seg = pd.read_csv(item, delimiter=',', header=None, dtype=str).values
    groups.extend([sub_idx] * len(audio_seg))
    sound_data.append(audio_seg)
sound_data = np.concatenate(sound_data, axis = 0)
features, labels = sound_data[:, :feat_size].astype(float), sound_data[:, -1]
groups = np.asarray(groups)


#%%
# keep only relevant instances (2 and m)
idx_voice = np.where(labels == '2')
idx_mixed = np.where(labels == 'm')
labels_voice = labels[idx_voice]
labels_mixed = labels[idx_mixed]
labels = np.concatenate((labels_voice, labels_mixed))
features_voice = features[idx_voice]
features_mixed = features[idx_mixed]
features = np.concatenate((features_voice, features_mixed))
groups_voice = groups[idx_voice]
groups_mixed = groups[idx_mixed]
groups = np.concatenate((groups_voice, groups_mixed))

idx_mixed = np.where(labels == 'm')   
labels[idx_mixed] = 0
idx_voice = np.where(labels == '2')   
labels[idx_voice] = 1

labels = labels.astype(int) 

#%%
# get a contextual instance based on each unit instance
t= 10
features_new = np.empty((0,1000))
groups_new = np.empty((0,1))
labels_new = np.empty((0,1))
for i in range(0, len(labels)-t, t):
    # only mean the instances within the same groups(subjects) and ignore ones of the transition
    if groups[i] == groups[i+t]:        
        temp_feat = features[i:i+t]
        # mean for every t seconds
        mean_feat = np.mean(temp_feat, axis=0)
        #mean_feat = temp_feat
        mean_feat = np.reshape(mean_feat, (1,1000))
    else:
        pass
    features_new = np.vstack((features_new, mean_feat))
    labels_new = np.vstack((labels_new, labels[i]))
    groups_new = np.vstack((groups_new, groups[i]))
labels_new = labels_new.astype(int)
groups_new = groups_new.reshape((groups_new.shape[0], ))
groups_new = groups_new.astype(int)

#labels_new = labels
#groups_new = groups
#features_new = features
#%%
RF = True
if RF == True:
    seed(0)
    predict = True   # True, False
    save_model_path = './models/RF/'
    check_dirs.check_dir(save_model_path)
    #kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    lppo = LeavePGroupsOut(n_groups=1)
    fold_no = 1
    f1_per_fold, acc_per_fold, pre_per_fold, rec_per_fold = [], [], [], []
    features = np.reshape(features, (features.shape[0], features.shape[1]))
    # training
    if not predict:
        #for train, test in kfold.split(features, labels):
        for train, test in lppo.split(features_new, labels_new, groups=groups_new):
              feat_train, labels_train = features_new[train], labels_new[train].reshape((len(labels_new[train])))
              feat_test, labels_test = features_new[test], labels_new[test].reshape((len(labels_new[test])))
              clf = RandomForestClassifier(n_estimators=100, random_state = 0, n_jobs=-1)
              # Fit data to model, then save models
              clf.fit(feat_train, labels_train)
              para = clf.get_params()
              # name: RandomForestClassifier_fold%d_estimators_%d
              filename =  'fold%d_' %fold_no + str(clf).split('(')[0] + '_estimators_%d' %para['n_estimators']
              joblib.dump(clf, save_model_path + filename)
              fold_no = fold_no + 1
    # validation                     
    if predict:
          models = sorted([os.path.join(save_model_path, item) \
                for item in os.listdir(save_model_path)], key=last_4chars2)
          #for train, test in kfold.split(features, labels):
          for train, test in lppo.split(features_new, labels_new, groups=groups_new):
              feat_train, labels_train = features_new[train], labels_new[train]
              feat_test, labels_test = features_new[test], labels_new[test]
              model_pred = joblib.load(models[fold_no - 1])
              # prediction
              pred = model_pred.predict(feat_test)
#              # set threshold for binary classification
#              if mode == '2class':
#                  idx_p = np.where(pred > 0.5)
#                  idx_n = np.where(pred <= 0.5)
#                  pred[idx_p] = 1
#                  pred[idx_n] = 0
#              elif mode == '3class':
#                  pass
              acc_per_fold.append(balanced_accuracy_score(labels_test, pred) * 100)
              f1_per_fold.append(f1_score(labels_test, pred, average = 'macro') * 100) 
              # initialize
              del model_pred
              fold_no = fold_no + 1
          f1 = np.mean(f1_per_fold)
          acc = np.mean(acc_per_fold)

          print('f1: ', f1, '\n acc: ', acc)
#%%
seed(0)
# keras seed
set_seed(0)
#features_new = np.expand_dims(features_new, axis=-1).astype(float)
#%%
          
NN = False
if NN:
    predict = False
    save_model_path = './models/'
    check_dirs.check_dir(save_model_path)
    batch_size = 128
    lppo = LeavePGroupsOut(n_groups=1)
    #lppo = GroupKFold(n_splits=2)
    #kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    fold_no = 1
    f1_per_fold, acc_per_fold, pre_per_fold, rec_per_fold = [], [], [], []
    # training
    if not predict:
        #for train, test in kfold.split(features, labels):
        for train, test in lppo.split(features_new, labels_new, groups_new):
              feat_train, labels_train = features_new[train], labels_new[train]
              feat_test, labels_test = features_new[test], labels_new[test]
              model_fold = model_nn(input_shape=(t, feat_size, 1)).model()

              # Fit data to model, only models with the best val acc (unbalanced) are saved
              model_fold.fit(feat_train, labels_train,
                          batch_size=batch_size,
                          epochs=5,
                          validation_data=(feat_test, labels_test),
                          verbose=2,
                          callbacks=[K.callbacks.ModelCheckpoint(save_model_path+"fold%01d-epoch_{epoch:02d}-acc_{val_binary_accuracy:.4f}.h5" %fold_no, 
                                                                 monitor='val_binary_accuracy', 
                                                                 verbose=0, 
                                                                 save_best_only=True, 
                                                                 save_weights_only=True, 
                                                                 mode='auto', 
                                                                 save_freq='epoch')])
              del model_fold
              fold_no = fold_no + 1
    # validation                     
    if predict:
          models = sorted([os.path.join(save_model_path, item) \
                    for item in os.listdir(save_model_path) if item.endswith('.h5')], key=last_4chars)
          #for train, test in kfold.split(features, labels):
          for train, test in lppo.split(features_new, labels_new, groups=groups_new):
              feat_train, labels_train = features_new[train], labels_new[train]
              feat_test, labels_test = features_new[test], labels_new[test]
              model_pred = model_nn(input_shape=(t, feat_size, 1)).model()

              model_pred.load_weights(models[fold_no - 1])
              # prediction
              pred_prob = model_pred.predict(feat_test)
              # set threshold for binary classification
              idx_p = np.where(pred_prob > 0.5)
              idx_n = np.where(pred_prob <= 0.5)
              pred_prob[idx_p] = 1
              pred_prob[idx_n] = 0

              acc_per_fold.append(balanced_accuracy_score(labels_test, pred_prob) * 100)
              f1_per_fold.append(f1_score(labels_test, pred_prob, average = 'macro') * 100)  
              # initialize
              del model_pred
              fold_no = fold_no + 1
          f1 = np.mean(f1_per_fold)
          acc = np.mean(acc_per_fold)
          print('acc: ', acc, '\n f1: ', f1)