"""
Author: Ivo Gollini Navarrete
Date: 24/01/2022
"""

import numpy as np
import os
import cv2 as cv
import skimage
import itertools
from scipy.stats import kurtosis, skew

def laplace_of_gaussian(gray_img, sigma, kappa=0.75, pad=False):
    """
    From: https://stackoverflow.com/questions/22050199/python-implementation-of-the-laplacian-of-gaussian-edge-detection
    Applies Laplacian of Gaussians to grayscale image.

    :param gray_img: image to apply LoG to
    :param sigma:    Gauss sigma of Gaussian applied to image, <= 0. for none
    :param kappa:    difference threshold as factor to mean of image values, <= 0 for none
    :param pad:      flag to pad output w/ zero border, keeping input image size
    """
    assert len(gray_img.shape) == 2
    img = cv.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
    img = cv.Laplacian(img, cv.CV_64F)
    rows, cols = img.shape[:2]
    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    # bool matrix for image value positiv (w/out border pixels)
    pos_img = 0 < img[1:rows-1, 1:cols-1]
    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    # sign change at pixel?
    zero_cross = neg_min + pos_max
    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    # optional thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.
    log_img = values.astype(np.uint8)
    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)
    
    return log_img


def log_filter(slice, sigmas):
    filtered_list = []

    for sigma in sigmas:
        filtered_frame = laplace_of_gaussian(slice, sigma)
        filtered_list.append(filtered_frame)
    filtered_list = np.dstack(filtered_list)

    return np.transpose(filtered_list, (2,0,1))

def glcm_extract_features(slices, degrees):
    glcm_features_list = []

    for i in range(len(slices)):
        glc_matrix = skimage.feature.graycomatrix(slices[i], [1], degrees) # Gray Level Co-ocurrence Matrix
        contrast = skimage.feature.graycoprops(glc_matrix, 'contrast')
        correlation = skimage.feature.graycoprops(glc_matrix, 'correlation')
        entropy = skimage.feature.graycoprops(glc_matrix, 'ASM')
        homogeneity = skimage.feature.graycoprops(glc_matrix, 'homogeneity')
        energy = skimage.feature.graycoprops(glc_matrix, 'energy')

        feat_list = [contrast[0], correlation[0], entropy[0], homogeneity[0], energy[0]]
        feat_list = list(itertools.chain.from_iterable(feat_list)) # 20 features (contrast 1,2,3,4; correlation 1,2,3,4: entropy...)
        glcm_features_list.append(feat_list)

    glcm_features_list = list(itertools.chain.from_iterable(glcm_features_list)) # 100 features (20 feat filt1, 20 feat filt2, 20 feat...)
    
    return glcm_features_list

def histogram_extract_features(slices):
    hist_features_list = []

    for i in range(len(slices)):
        per1 = np.percentile(slices[i], 10, axis=0)
        per2 = np.percentile(slices[i], 25, axis=0)
        per3 = np.percentile(slices[i], 50, axis=0)

        mean_val = np.mean(slices[i])
        mean_per_0 = np.mean(per1)
        mean_per_1 = np.mean(per2)
        mean_per_2 = np.mean(per3)
        # print(mean_val, mean_per_0, mean_per_1, mean_per_2)

        std_val = np.std(slices[i])
        std_per_0 = np.std(per1)
        std_per_1 = np.std(per2)
        std_per_2 = np.std(per3)
        # print(std_val, std_per_0, std_per_1, std_per_2)

        kurtosis_val = kurtosis(slices[i].flatten())
        skew_val = skew(slices[i].flatten())
        # print(kurtosis_val, skew_val)

        hist_features_list.append([
            mean_val, mean_per_0, mean_per_1, mean_per_2,
            std_val, std_per_0, std_per_1, std_per_2,
            kurtosis_val, skew_val]
            ) # 10 features
    
    hist_features_list = list(itertools.chain.from_iterable(hist_features_list)) # 50 features (10 feat filt1, 10 feat filt2, 10 feat...)
    
    return hist_features_list


def manual_feat_extraction(in_dir):
    log_sigmas = [0, 1, 1.5, 2, 2.5] # Laplace of Gaussian (LoG) sigmas
    glcm_deg = [0, np.pi/4, np.pi/2, np.pi*3/4] # Gray Level Co-ocurrence Matrix (GLCM) [0, 45, 90, 135]

    paths_ROI = sorted(os.listdir(in_dir))

    feat_list = []
    for roi in paths_ROI:
        roi_vol = np.load(os.path.join(in_dir, roi))
        print(roi, roi_vol.shape)
        roi_vol = np.transpose(roi_vol, (2,0,1)) # Iterate over frames dimension
        
        features_roi = []
        for frame in range(len(roi_vol)):
            filtered_frame = log_filter(roi_vol[frame], log_sigmas)
            # print(filtered_frame.shape) # (5, 222, 222)
            feat_glcm = glcm_extract_features(filtered_frame, glcm_deg)
            # print(len(feat_glcm)) # 100
            feat_hist = histogram_extract_features(filtered_frame)
            # print(len(feat_hist)) # 50
            manual_features = feat_glcm + feat_hist
            # print(len(manual_features)) # 150 
            features_roi.append(manual_features)
        
        features_roi = np.stack(features_roi)
        # print('Features shape:', features_roi.shape)
        # feat_name = roi[:8] + 'ManFeat'
        # np.save(os.path.join('data/Manual_Features', feat_name), features_roi)

        feat_list.append(features_roi.flatten())

    feat_list = np.array(feat_list)

    # np.save(os.path.join('data', feat_name), features_roi)
    print('Manual Features shape:', feat_list.shape)

    return feat_list

# in_path = '/home/ivo.navarrete/Desktop/NSCLC/radiogenomics/data/ROIs_3frames/3channels/resized_224'
# manual_feat_extraction(in_path)