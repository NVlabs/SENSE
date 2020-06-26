"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

def eval_flow(flow_pred, flow_gt, mask=None):
    metrics = {}
    F_gt_du = flow_gt[:,:,0]
    F_gt_dv = flow_gt[:,:,1]
    F_est_du = flow_pred[:,:,0]
    F_est_dv = flow_pred[:,:,1]
    E_du = F_gt_du - F_est_du
    E_dv = F_gt_dv - F_est_dv
    f_gt_norm = np.sqrt(np.square(F_gt_du) + np.square(F_gt_dv))

    valid = (f_gt_norm != 0).astype(int)
    E = np.sqrt(np.square(E_du) + np.square(E_dv))
    E[valid==0] = 0

    metrics['valid'] = valid
    metrics['px_sum'] = np.sum(valid)
    metrics['e_sum'] = np.sum(E)
    percent_err = E / (1e-12 + f_gt_norm)
    metrics['percent_err_map'] = percent_err
    metrics['error_map'] = E
    # calculate bad point in KITTI flow evaluation
    bad_point = (E >= 3) * (percent_err >= .05)
    metrics['bad_map'] = bad_point
    metrics['bad_point'] = np.sum(bad_point)
    metrics['epe'] = float(metrics['e_sum']) / metrics['px_sum']
    metrics['percent_bad'] = float(metrics['bad_point']) / metrics['px_sum']

    if mask is None:
        return metrics
        
    mask_fg = (mask != 0)
    mask_bg = (mask == 0)
    metrics['px_fg'] = np.sum(valid * mask_fg)
    metrics['px_bg'] = np.sum(valid * mask_bg)
    metrics['e_fg'] = np.sum(E * mask_fg)
    metrics['e_bg'] = np.sum(E * mask_bg)
    metrics['bad_fg'] = np.sum(bad_point * mask_fg)
    metrics['bad_bg'] = np.sum(bad_point * mask_bg)
    metrics['epe_fg'] = float(metrics['e_fg']) / (1e-12 + metrics['px_fg'])
    metrics['epe_bg'] = float(metrics['e_bg']) / (1e-12 + metrics['px_bg'])
    metrics['percent_fg'] = float(metrics['bad_fg']) / (1e-12 + metrics['px_fg'])
    metrics['percent_bg'] = float(metrics['bad_bg']) / (1e-12 + metrics['px_bg'])
    
    return metrics

def eval_disp(disp_pred, disp_gt, mask=None):
    metrics = {}
    valid = (disp_gt != 0).astype(int)
    E = np.abs(disp_gt - disp_pred)
    E[valid==0] = 0
    metrics['px_sum'] = np.sum(valid)
    metrics['e_sum'] = np.sum(E)
    percent_err = E / (1e-12 + disp_gt)
    bad_point = (E >= 3) * (percent_err >= .05)
    metrics['valid'] = valid
    metrics['bad_map'] = bad_point
    metrics['percent_err_map'] = percent_err
    metrics['error_map'] = E
    metrics['bad_point'] = np.sum(bad_point)
    metrics['epe'] = float(metrics['e_sum']) / metrics['px_sum']
    metrics['percent_bad'] = float(metrics['bad_point']) / metrics['px_sum']

    if mask is None:
        return metrics
    mask_fg = (mask != 0)
    mask_bg = (mask == 0)
    metrics['px_fg'] = np.sum(valid * mask_fg)
    metrics['px_bg'] = np.sum(valid * mask_bg)
    metrics['e_fg'] = np.sum(E * mask_fg)
    metrics['e_bg'] = np.sum(E * mask_bg)
    metrics['bad_fg'] = np.sum(bad_point * mask_fg)
    metrics['bad_bg'] = np.sum(bad_point * mask_bg)
    metrics['epe_fg'] = float(metrics['e_fg']) / (1e-12 + metrics['px_fg'])
    metrics['epe_bg'] = float(metrics['e_bg']) / (1e-12 + metrics['px_bg'])
    metrics['percent_fg'] = float(metrics['bad_fg']) / (1e-12 + metrics['px_fg'])
    metrics['percent_bg'] = float(metrics['bad_bg']) / (1e-12 + metrics['px_bg'])
    return metrics
