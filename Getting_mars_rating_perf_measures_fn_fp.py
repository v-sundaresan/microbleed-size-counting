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
subjs = glob.glob('./Data/*/mni_2swi.mat')
subjs = glob.glob('./Predictions/Test_final_labs_*.npy')
conf_matrices = []
red_conf_matrices = []
sensitivities = np.zeros([len(subjs),3])
precisions = np.zeros([len(subjs),3])
f1_measures = np.zeros([len(subjs),3])
accuracies = np.zeros([len(subjs),2])
count_allstruct_array_misses = np.zeros([13,len(subjs)])
count_allstruct_array_fps = np.zeros([13,len(subjs)])
for i in range(len(subjs)):
    prediction = np.load(subjs[i])
    gt = np.round(nib.load('/vols/Scratch/vaanathi/Rob_data/original/labels/' + subjs[i][78:-4] + '_SWI_CMB.nii.gz').get_data())
    atlas = np.round(nib.load('/vols/Scratch/vaanathi/CMB_deep_learning/Raw_Rob_data/' + subjs[i][78:-4] + '/MARS_scale_parcellation_map_org.nii.gz').get_data())
    labgt, nlabgt = label(gt>0, return_num=True)
    dellabs = []
    for lb in range(nlabgt):
        predvxls = prediction[labgt == lb+1]
        if np.sum(predvxls) > 0:
            dellabs.append(lb+1)
        
    for dl in dellabs:
        labgt[labgt==dl] = 0

    gt = gt * (labgt> 0).astype(float)

    labpred, nlabpred = label(prediction>0, return_num = True)
    dellabs = []
    for lb in range(nlabpred):
        gtvxls = gt[labpred ==lb+1]
        if np.sum(gtvxls)>0:
            dellabs.append(lb+1)

    for dl in dellabs:
        labpred[labpred==dl] = 0

    print(np.union1d(atlas,[]))


    pred = labpred>0
    labpred = label(pred)
    gtprops = regionprops(labpred)
    gtcents = cents = np.array([np.round(p.centroid) for p in gtprops])
    for c in range(gtcents.shape[0]):
        col_id = atlas[int(gtcents[c][0]), int(gtcents[c][1]), int(gtcents[c][2])]-1
        try:
            count_allstruct_array_fps[int(col_id), i] += 1
        except:
            count_allstruct_array_fps[int(np.sqrt(col_id)), i] += 1
    for ind in range(13):
        def_set = [(ind*4)+1,(ind*4)+2]
        for st in def_set:
            gt_map = gt == st
            labelmap, n_lab = label(gt_map,return_num=True)
            props = regionprops(labelmap)
            cents = np.array([np.round(p.centroid) for p in props])
            for c in range(cents.shape[0]):
                col_id = atlas[int(cents[c][0]), int(cents[c][1]), int(cents[c][2])]-1
                try:
                    count_allstruct_array_misses[int(col_id), i] += 1
                except:
                    count_allstruct_array_misses[int(np.sqrt(col_id)), i] += 1
np.savez('./Data/MARS_evaluation_metrics_fps_fns',misses=count_allstruct_array_misses, fps=count_allstruct_array_fps)
        
        
        

    

    
    
    


