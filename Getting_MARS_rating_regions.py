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

for i in range(len(subjs)):
    brain = (nib.load(subjs[:-13] + '_SWIresampled_111mm_brain_mask.nii.gz') > 0).get_data().astype(float)
    brainstem = (nib.load(subjs[:-12] + 'brainstem_lin.nii.gz') > 0).get_data().astype(float)
    cerebellum = (nib.load(subjs[:-12] + 'cerbellum_lin.nii.gz') > 0).get_data().astype(float)
    basal_ganglia = (nib.load(subjs[:-12] + 'basal_ganglia_lin.nii.gz') > 0).get_data().astype(float)
    thalamus = (nib.load(subjs[:-12] + 'thalamus_lin.nii.gz') > 0).get_data().astype(float)
    int_capsule = (nib.load(subjs[:-12] + 'int_capsule_lin.nii.gz') > 0).get_data().astype(float)
    ext_capsule = (nib.load(subjs[:-12] + 'ext_capsule_lin.nii.gz') > 0).get_data().astype(float)
    corpus_callosum = (nib.load(subjs[:-12] + 'ext_capsule_lin.nii.gz') > 0).get_data().astype(float)
    wm = (nib.load(subjs[:-12] + 'white_matter_lin.nii.gz').get_data() > 0).astype(float)
    frontal = (nib.load(subjs[:-12] + 'frontal_lin.nii.gz').get_data() > 0).astype(float)
    parietal = (nib.load(subjs[:-12] + 'parietal_lin.nii.gz').get_data() > 0).astype(float)
    temporal = (nib.load(subjs[:-12] + 'temporal_lin.nii.gz').get_data() > 0).astype(float)
    occipital = (nib.load(subjs[:-12] + 'occipital_lin.nii.gz').get_data() > 0).astype(float)
    insular = (nib.load(subjs[:-12] + 'insular_lin.nii.gz').get_data() > 0).astype(float)

    kernel_size = 3

    insular_dilated_mask = binary_dilation(insular>0, structure=None, iterations=kernel_size)
    frontal_dilated_mask = binary_dilation(frontal>0, structure=None, iterations=kernel_size)
    parietal_dilated_mask = binary_dilation(parietal>0, structure=None, iterations=kernel_size)
    occipital_dilated_mask = binary_dilation(occipital>0, structure=None, iterations=kernel_size)
    temporal_dilated_mask = binary_dilation(temporal>0, structure=None, iterations=kernel_size)

    inpaint_base = np.zeros(brainstem.shape)
    inpaint_base[frontal_dilated_mask > 0] = 9
    inpaint_base[parietal_dilated_mask > 0] = 10
    inpaint_base[temporal_dilated_mask > 0] = 11
    inpaint_base[occipital_dilated_mask > 0] = 12
    inpaint_base[insular_dilated_mask > 0] = 13

    all_structs = np.zeros(brainstem.shape)
    all_structs[wm>0] = 8
    all_structs[brainstem>0] = 1
    all_structs[cerebellum>0] = 2
    all_structs[basal_ganglia>0] = 3
    all_structs[thalamus>0] = 4
    all_structs[int_capsule>0] = 5
    all_structs[ext_capsule>0] = 6
    all_structs[corpus_callosum>0] = 7
    all_structs[frontal > 0] = 9
    all_structs[parietal > 0] = 10
    all_structs[temporal > 0] = 11
    all_structs[occipital > 0] = 12
    all_structs[insular > 0] = 13

    all_structs_final = np.zeros(brainstem.shape)
    all_structs_final[inpaint_base == 9] = 9
    all_structs_final[inpaint_base == 10] = 10
    all_structs_final[inpaint_base == 11] = 11
    all_structs_final[inpaint_base == 12] = 12
    all_structs_final[inpaint_base == 13] = 13
    all_structs_final[all_structs == 8] = 8
    all_structs_final[all_structs == 1] = 1
    all_structs_final[all_structs == 2] = 2
    all_structs_final[all_structs == 3] = 3
    all_structs_final[all_structs == 4] = 4
    all_structs_final[all_structs == 5] = 5
    all_structs_final[all_structs == 6] = 6
    all_structs_final[all_structs == 7] = 7
    all_structs_final[brain == 0] = 0

    brain = nib.load(subjs[:-13] + '_SWIresampled_111mm_brain_mask.nii.gz')
    bheader = brain.header
    newhdr = bheader.copy()
    newobj_mars = nib.nifti1.Nifti1Image(all_structs_final, None, header=newhdr)
    nib.save(newobj_mars, subjs[:-12] + 'MARS_scale_parcellation_map.nii.gz')

    

    
    
    


