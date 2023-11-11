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

brain = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/MNI152_T1_1mm_brain.nii.gz').get_data() > 0).astype(float)
brainstem = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/brainstem_bin.nii.gz').get_data() > 0).astype(float)
cerebellum = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/cerbellum_bin_mni.nii.gz').get_data() > 0).astype(float)
basal_ganglia = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/basal_ganglia_bin.nii.gz').get_data() > 0).astype(float)
thalamus = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/thalamus_bin.nii.gz').get_data() > 0).astype(float)
int_capsule = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/internal_capsule_bin.nii.gz').get_data() > 0).astype(float)
ext_capsule = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/external_capsule_bin.nii.gz').get_data() > 0).astype(float)
corpus_callosum = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/corpus_callosum_bin.nii.gz').get_data() > 0).astype(float)
wm = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/white_matter_bin.nii.gz').get_data() > 0).astype(float)
frontal = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/frontal_lobe_bin.nii.gz').get_data() > 0).astype(float)
parietal = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/parietal_lobe_bin.nii.gz').get_data() > 0).astype(float)
temporal = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/temporal_lobe_bin.nii.gz').get_data() > 0).astype(float)
occipital = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/occipital_lobe_bin.nii.gz').get_data() > 0).astype(float)
insular = (nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/insular_lobe_bin.nii.gz').get_data() > 0).astype(float)

insular_dilated_mask = binary_dilation(insular>0, structure=None, iterations=3)
frontal_dilated_mask = binary_dilation(frontal>0, structure=None, iterations=5)
parietal_dilated_mask = binary_dilation(parietal>0, structure=None, iterations=7)
occipital_dilated_mask = binary_dilation(occipital>0, structure=None, iterations=2)
temporal_dilated_mask = binary_dilation(temporal>0, structure=None, iterations=2)

inpaint_base = np.zeros(brainstem.shape)
inpaint_base[insular_dilated_mask > 0] = 13
inpaint_base[frontal_dilated_mask > 0] = 9
inpaint_base[occipital_dilated_mask > 0] = 12
inpaint_base[parietal_dilated_mask > 0] = 10
inpaint_base[temporal_dilated_mask > 0] = 11

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

brain = nib.load('/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/MNI152_T1_1mm_brain.nii.gz')
bheader = brain.header
newhdr = bheader.copy()
newobj_mars = nib.nifti1.Nifti1Image(all_structs_final, None, header=newhdr)
nib.save(newobj_mars, '/vols/Scratch/vaanathi/MARS_rating_analysis/Atlases/MARS_scale_parcellation_map_mni.nii.gz')

    

    
    
    


