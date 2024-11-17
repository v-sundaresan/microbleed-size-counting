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
'''
brainstem_def = [1,2]
cerebellum_def = [5,6]
basal_ganglia_def = [9,10]
thalamus_def = [13,14]
int_capsule_def = [17,18]
ext_capsule_def = [21,22]
corpus_callosum_def = [25,26]
wm_def = [29,30]
frontal_def = [33,34]
parital_def = [37,38]
occipital_def = [45,46]
temporal_def = [41,42]
insula_def = [49,50]
brainstem_pos = [3,4]
cerebellum_pos = [7,8]
basal_ganglia_pos = [11,12]
thalamus_pos = [15,16]
int_capsule_pos = [19,20]
ext_capsule_pos = [23,24]
corpus_callosum_pos = [27,28]
wm_pos = [31,32]
frontal_pos = [35,36]
parital_pos = [39,40]
occipital_pos = [47,48]
temporal_pos = [43,44]
insula_pos = [51,52]
'''
mars_values_gt_map_possible = []
mars_values_gt_map_definite = []
for i in range(len(subjs)):
    mars_gt_def_values = np.zeros([13,1])
    mars_gt_pos_values = np.zeros([13,1])
    gt = np.round(nib.load('./Labels' + subjs[i][54:-13] + '_SWI_CMB.nii.gz').get_data())
    
    for ind in range(13):
        def_set = [(ind*4)+1,(ind*4)+2]
        pos_set = [(ind*4)+3,(ind*4)+4]
        temp_count = []
        for st in def_set:
            gt_map = gt == st
            labelmap, n_lab = label(gt_map,return_num=True)
            temp_count.append(n_lab)
        total_count = np.sum(np.array(temp_count))
        mars_gt_def_values[ind] = total_count

        temp_count = []
        for st in pos_set:
            gt_map = gt == st
            labelmap, n_lab = label(gt_map,return_num=True)
            temp_count.append(n_lab)
        total_count = np.sum(np.array(temp_count))
        mars_gt_pos_values[ind] = total_count
    mars_values_gt_map_definite.append(mars_gt_def_values)
    mars_values_gt_map_possible.append(mars_gt_pos_values)
    np.save(subjs[i][:-12] + 'MARS_scale_gt_def_values',mars_gt_def_values)
    np.save(subjs[i][:-12] + 'MARS_scale_gt_pos_values',mars_gt_pos_values)
np.save('./Data/MARS_rating_gt_def_map',mars_values_gt_map_definite)       
np.save('./Data/MARS_rating_gt_pos_map',mars_values_gt_map_possible) 


        
        

    

    
    
    


