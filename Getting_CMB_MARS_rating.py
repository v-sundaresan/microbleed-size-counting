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

mars_values_map = np.zeros([12,len(subjs)])
for i in range(len(subjs)):
    lab = (nib.load(subjs[i][:-13] + '_SWI_CMBresampled_111mm.nii.gz').get_data() > 0).astype(float)
    atlas = nib.load(subjs[i][:-12] + 'MARS_scale_parcellation_map.nii.gz').get_data()
    label_lab, n_lab = label(lab > 0, return_num=True)
    mars_values = np.zeros([12,1])
    for st in range(13,0,-1):
        struct_mask = (atlas == st).astype(float)
        labels = label_lab[struct_mask>0]
        labels_found = np.setdiff2d(np.union1d(labels,[]),[0])
        mars_values[st-1] = len(labels_found)
        for l in labels_found:
            label_lab[label_lab == l] = 0
        label_lab, n_lab = label(label_lab > 0, return_num=True)
    mars_values_map[:,i] = mars_values
    np.save(subjs[i][:-12] + 'MARS_scale_values',mars_values)
np.save('/vols/Scratch/vaanathi/CMB_deep_learning/Raw_Rob_data/MARS_rating_map',mars_values_map)
        
        
        

    

    
    
    


