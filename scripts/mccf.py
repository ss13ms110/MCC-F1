# Author: Shubham Sharma (https://github.com/ss13ms110)
# MCC-F1 analysis
# Python version of MSS-F1 analysis of https://github.com/hoffmangroup/mccf1/
# input: normalied MCC and F1 file in below format
#       MCC_Normalized      F1      CFS
#       0.472641        0.091178  -6.90
#       0.476918        0.091188  -6.69
#       0.468482        0.090971  -6.62
#       ...             ...        ...
# output: MCC-F1 metric
# figures: MCC-F1 figures stored in ./figs

import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# FUNCTIONS ===========================================================
# make plot ------
def plot_mcc(f1, mcc_nor, fig_name):
    plt.figure(figsize=(8,8))
    plt.xlabel('F1 score', fontsize=20)
    plt.ylabel('normalized MCC', fontsize=20)
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.title('MCC-F1 curve for %s' %(fig_name.split('/')[-1].split('.')[0]))
    plt.scatter([0,1], [0,1], c=['red', 'green'], s=2**6)
    plt.plot(f1, mcc_nor, color='black', linewidth=1)
    plt.savefig(fig_name)

# calculate distance ------
def calc_mean_dist(x1, y1, x2, y2):
    return np.mean(np.sqrt((x1-x2)**2 + (y1-y2)**2))

# ====================================================================
# main
mcc_input_dir = './inputs'
fig_dir = './figs'
mcc_output_file = './outputs/mcc-f1.out'

# params
bins = 100

# output
mcc_file_list = os.listdir(mcc_input_dir)
out_file = open(mcc_output_file, 'w')
out_file.write('input-file          mcc-f1-metric\n')

# loop in files
for mcc_file in mcc_file_list:
    print('\n    Working on %s  '%(mcc_file))
    # load file data
    mcc_file_path = '%s/%s' %(mcc_input_dir, mcc_file)
    mccNor_f1_cfs = np.loadtxt(mcc_file_path)

    # get rid of NaN values in MCC and F1
    mccNorF1_truncated = mccNor_f1_cfs[~np.isnan(mccNor_f1_cfs).any(axis=1)]

    # --------------------------------------
    # make plot
    # --------------------------------------
    fig_name = '%s/%s.png' %(fig_dir, mcc_file.split('.')[0])
    plot_mcc(mccNorF1_truncated[:,1], mccNorF1_truncated[:,0], fig_name)
    # ======================================

    # --------------------------------------
    # calculate MCC-F1 metric
    # --------------------------------------

    # get MCC and F1 truncated out of the matrix
    mccNor, f1 = mccNorF1_truncated[:,0], mccNorF1_truncated[:,1]

    # get index of maximum MCC
    max_mc_index = np.argmax(mccNor)

    # divide the MCC-F1 curve in two parts
    # before the max MCC and after the max MCC

    # ---------- before
    mcc_before = mccNor[:max_mc_index]
    f_before = f1[:max_mc_index]

    # ---------- after
    mcc_after = mccNor[max_mc_index:]
    f_after = f1[max_mc_index:]

    # get length of bin
    unit_len = (max(mccNor) - min(mccNor))/bins

    # Now in each bin (remember these bins are vertical) calculate the mean distance of MCC-F1
    # points to the point of perfection (1,1). Then, take average of all mean points
    mean_dist_before, mean_dist_after = [], []
    for i in range(bins):
        # before
        mcc_indx_before = ((mcc_before >= min(mccNor) + i*unit_len) & (mcc_before <= min(mccNor) + (i+1)*unit_len))
        mean_dist_before = np.append(mean_dist_before, calc_mean_dist(f_before[mcc_indx_before], mcc_before[mcc_indx_before], 1, 1))
        
        # after
        mcc_indx_after = ((mcc_after >= min(mccNor) + i*unit_len) & (mcc_after <= min(mccNor) + (i+1)*unit_len))
        mean_dist_after = np.append(mean_dist_after, calc_mean_dist(f_after[mcc_indx_after], mcc_after[mcc_indx_after], 1, 1))

    # club all distances in one array
    all_mean_distances = np.concatenate((mean_dist_before,mean_dist_after))

    # drop NaN
    all_mean_distances_no_NaN = all_mean_distances[~np.isnan(all_mean_distances)]

    # take average
    distance_average = np.mean(all_mean_distances_no_NaN)

    # calculate MCC-F1 metric
    MCC_F1_metric = 1 - distance_average/np.sqrt(2)

    out_file.write('%s      %7.5f\n' %(mcc_file, MCC_F1_metric))
    # ======================================
print('\n           DONE!!!\n Figures saved in %s\nOutput saved in %s\n' %(fig_dir, mcc_output_file))