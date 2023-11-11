#!/usr/bin/env fslpython
#   Copyright (C) 2016 University of Oxford 
#   SHBASECOPYRIGHT

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from skimage.measure import regionprops, label
from skimage.restoration import inpaint
from scipy.ndimage.morphology import binary_dilation
import glob

'''
Order of colours:
Brainstem = 1
cerebellum = 2
basalganglia = 3
Thalamus = 4
Intcapsule = 5
Extcapsule = 6
corpuscallosum = 7
wm = 8
frontal = 9
parietal = 10
temporal = 11
occipital = 12
insula = 13
'''
subjs = glob.glob('/vols/Scratch/vaanathi/CMB_deep_learning/Raw_Rob_data/*/mni_2swi.mat')

conf_matrices = []
red_conf_matrices = []
sensitivities = np.zeros([len(subjs),3])
precisions = np.zeros([len(subjs),3])
f1_measures = np.zeros([len(subjs),3])
accuracies = np.zeros([len(subjs),2])
for i in range(len(subjs)):
    gt = np.round(nib.load('/vols/Scratch/vaanathi/Rob_data/original/labels/' + subjs[i][54:-13] + '_SWI_CMB.nii.gz').get_data())
    atlas = np.round(nib.load(subjs[i][:-12] + 'MARS_scale_parcellation_map_org.nii.gz').get_data())
        for i in range(13):
        def_set = [(i*4)+1,(i*4)+2]
        for st in def_set:
            gt_map = gt == st
            labelmap, n_lab = label(gt_map,return_num=True)
            props = regionprops(labelmap)
            cents = np.array([np.round(p.centroid) for p in props])
            for c in range(cents.shape[0]):
                col_id = atlas[int(cents[c][0]), int(cents[c][1]), int(cents[c][2])]-1
                conf_matrix[i,int(col_id)] += 1
    red_conf_matrix = np.zeros([3,3])
    red_conf_matrix[0][0] = np.sum(conf_matrix[:2,:2])
    red_conf_matrix[1][1] = np.sum(conf_matrix[2:8,2:8])
    red_conf_matrix[2][2] = np.sum(conf_matrix[8:13,8:13])
    red_conf_matrix[0][1] = np.sum(conf_matrix[:2,2:8])
    red_conf_matrix[0][2] = np.sum(conf_matrix[:2,8:13])
    red_conf_matrix[1][0] = np.sum(conf_matrix[2:8,:2])
    red_conf_matrix[1][2] = np.sum(conf_matrix[2:8,8:13])
    red_conf_matrix[2][0] = np.sum(conf_matrix[8:13,:2])
    red_conf_matrix[2][1] = np.sum(conf_matrix[8:13,2:8])

    conf_matrices.append(conf_matrix)
    red_conf_matrices.append(red_conf_matrix)
   
    struct_acc = np.sum(np.diagonal(conf_matrix))/np.sum(conf_matrix)
    reg_acc = np.sum(np.diagonal(red_conf_matrix))/np.sum(red_conf_matrix)

    [inftent_sens, deep_sens, lobar_sens] = np.diagonal(red_conf_matrix/np.sum(red_conf_matrix, axis=1))
    [inftent_prec, deep_prec, lobar_prec] = np.diagonal(red_conf_matrix/np.sum(red_conf_matrix, axis=0))
    
    sens = np.array([inftent_sens, deep_sens, lobar_sens])
    prec = np.array([inftent_prec, deep_prec, lobar_prec])

    [inftent_f1, deep_f1, lobar_f1] = 2*((sens*prec)/(sens + prec))
            
    sensitivities[i,:] = sens
    precisions[i,:] = prec
    f1_measures[i,:] = np.array([inftent_f1, deep_f1, lobar_f1])
    accuracies[i,:] = np.array([struct_acc, reg_acc])

    np.savez(subjs[i][:-12] + 'MARS_rating_evaluation_measures', conf=conf_matrix, red_conf=red_conf_matrix, sensitivity=sens, precision=prec, f1=np.array([inftent_f1, deep_f1, lobar_f1]), accuracies=np.array([struct_acc, reg_acc]))

np.savez('/vols/Scratch/vaanathi/CMB_deep_learning/Raw_Rob_data/MARS_evaluation_metrics_aggregate',conf=conf_matrices, red_conf=red_conf_matrices, sensitivity=sensitivities, precision=precisions, f1=f1_measures, accuracies=accuracies)
        
        
        

    

    
    
    


