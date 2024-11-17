import numpy as np
import nibabel as nib
import glob
from skimage.measure import label, regionprops

#=========================================================================================
# CMB Size estimation function
# Vaanathi Sundaresan
# 17-10-2024, IISc, India
#=========================================================================================


def get_cmb_chars(gt):
    lab_gt = label(gt > 0)
    prop_gt = regionprops(lab_gt)
    gt_areas = np.array([p.area for p in prop_gt])
    gt_majorlength = np.array([p.major_axis_length for p in prop_gt])
    return gt_areas, gt_majorlength


gt_dir = './Labels/'
pred_dir = './Predictions/'
out_dir = './MARS_sizes/'
subjs = glob.glob(pred_dir + '*_pred_2MNI.nii.gz')

mars_values_gt_map_possible = []
mars_values_gt_map_definite = []
regions = ['bs', 'ce', 'bg', 'th', 'ic', 'ec', 'cc', 'wm', 'fl', 'pl', 'ol', 'tl', 'il']
mars_gt_size_def = []
mars_gt_size_pos = []
mars_gt_area_def = []
mars_gt_area_pos = []
for i in range(len(subjs)):
    mars_gt_def_values = np.zeros([13, 1])
    mars_gt_pos_values = np.zeros([13, 1])
    gt = np.round(nib.load(gt_dir + subjs[i][102:-17] + '_SWI_CMB_2MNI.nii.gz').get_fdata())
    #     print(np.union1d(gt[gt>0], []))
    size_def = {}
    size_pos = {}
    area_def = {}
    area_pos = {}
    for ind in range(13):
        def_set = [(ind * 4) + 1, (ind * 4) + 2]
        pos_set = [(ind * 4) + 3, (ind * 4) + 4]
        temp_count = []
        new_gt = np.zeros_like(gt)
        for st in def_set:
            new_gt += (gt == st).astype(float)
        area_cmb, size_cmb = get_cmb_chars(new_gt.astype(float))
        #         print(size_cmb, area_cmb)
        size_def[regions[ind]] = size_cmb
        area_def[regions[ind]] = area_cmb
        #         print(ind, size_def)
        total_count = np.sum(np.array(temp_count))
        mars_gt_def_values[ind] = total_count

        temp_count = []
        new_gt = np.zeros_like(gt)
        for st in def_set:
            new_gt += (gt == st).astype(float)
        area_cmb, size_cmb = get_cmb_chars(new_gt.astype(float))
        size_pos[regions[ind]] = size_cmb
        area_pos[regions[ind]] = area_cmb

        total_count = np.sum(np.array(temp_count))
        mars_gt_pos_values[ind] = total_count
    #     print(size_def)
    mars_gt_area_pos.append(area_pos)
    mars_gt_size_pos.append(size_pos)
    mars_gt_area_def.append(area_def)
    mars_gt_size_def.append(size_def)
    mars_values_gt_map_definite.append(mars_gt_def_values)
    mars_values_gt_map_possible.append(mars_gt_pos_values)
    np.save(out_dir + subjs[i][102:-17] + '_MARS_gt_area_pos', area_pos)
    np.save(out_dir + subjs[i][102:-17] + '_MARS_gt_size_pos', size_pos)
    np.save(out_dir + subjs[i][102:-17] + '_MARS_gt_area_def', area_def)
    np.save(out_dir + subjs[i][102:-17] + '_MARS_gt_size_def', size_def)
np.save(out_dir + 'MARS_gt_def_sizes', mars_gt_size_def)
np.save(out_dir + 'MARS_gt_def_areas', mars_gt_area_def)
np.save(out_dir + 'MARS_gt_pos_sizes', mars_gt_size_pos)
np.save(out_dir + 'MARS_gt_pos_areas', mars_gt_area_pos)
