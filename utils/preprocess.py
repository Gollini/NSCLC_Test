"""
Author: Ivo Gollini Navarrete
Date: 14/01/2022
"""

import os
import glob 
import shutil
from tqdm import tqdm
import dicom2nifti
import numpy as np
import pydicom as dicom
import monai
import torch
import torchvision
from torchvision.ops import masks_to_boxes
from skimage.transform import resize

def dcm2nii(in_path, out_path):
    count = 0
    for patient in glob.glob(in_path + '/R01*'):
        patient_id = os.path.basename(os.path.normpath(patient))
        for only_ct in glob.glob(patient + '/*'):
            scan_id = os.path.basename(os.path.normpath(only_ct))

            # Select only the Folders that with the CT scans and segmentations.
            # Should be 144 scans with segmentation.
            if (scan_id.find('NA-CT') != -1) or (scan_id.find('NA-T') != -1) or (scan_id.find('NA-XT') != -1):
                print(patient_id)
                print(scan_id)
                for folder_ct in glob.glob(only_ct + '/*'):
                    seg_ct = os.path.basename(os.path.normpath(folder_ct))
                    if (seg_ct.find('seg') != -1) or (seg_ct.find('ePAD') != -1):
                        print('SEGMENTATION:', seg_ct)
                        out_dir = out_path + '/mask'
                        file_name = patient_id + '_mask'
                        shutil.copy2(folder_ct + '/1-1.dcm', os.path.join(out_dir, file_name + '.dcm'))
                        # dicom2nifti.dicom_series_to_nifti(folder_ct, os.path.join(out_dir, file_name + '.nii.gz'), reorient_nifti=True)
                        # dicom2nifti.convert_directory(folder_ct, os.path.join(out_dir, file_name + '.nii.gz'))
                        count += 1
                    
                    else: 
                        print('CT SCAN:', seg_ct)
                        out_dir = out_path + '/CT'
                        file_name = patient_id + '_CT'
                        print('Converting to NIFTI...', file_name)
                        try: dicom2nifti.dicom_series_to_nifti(folder_ct, os.path.join(out_dir, file_name + '.nii.gz'))
                        except:
                            print('CAREFUL with file:', patient_id)
                            continue
                        print('Converted\n')

                print(count, '\n')


def bbx_from_mask(seg_):
    frames_list = []
    max_mask_vol = 0
    max_mask_vol_frame = 0

    for frame in range(len(seg_)):
        if seg_[frame].max() > 0:
            mask_vol = np.count_nonzero(seg_[frame])
            if mask_vol > max_mask_vol:
                max_mask_vol = mask_vol
                max_mask_vol_frame = frame
            frames_list.append(frame)
    
    seg_bbx = seg_.copy()
    seg_bbx = seg_bbx[max_mask_vol_frame]
    seg_bbx = torch.Tensor(seg_bbx)[None,:,:] 
    bbx = masks_to_boxes(seg_bbx) # (xmin, ymin, xmax, ymax)
    bbx = bbx.numpy()
    # bbx = [int(x) for x in bbx[0]]

    return frames_list, max_mask_vol_frame, bbx

def norm_vol(vol):
    norm = np.clip(vol, -1000, 400)
    norm = (norm-norm.min())
    norm = (norm/norm.max())*255
    return norm

def volume_from_bbx(vol, frames, coord):
    tumor_segmentation = []
    vol_norm = norm_vol(vol)
    for i in frames:
        tumor_frame = vol_norm[i].copy()
        roi_frame = tumor_frame[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
        tumor_segmentation.append(roi_frame)

    roi_vol = np.dstack(tumor_segmentation)
    roi_resized = roi_vol.copy()
    roi_resized_112 = resize(roi_resized, (112, 112, roi_vol.shape[2]))
    roi_resized_64 = resize(roi_resized, (64, 64, roi_vol.shape[2]))

    return roi_vol, roi_resized_112, roi_resized_64

def tumor_roi(in_dir):
    path_volumes = sorted(os.listdir(in_dir + "/CT/"))
    path_segmentation = sorted(os.listdir( in_dir + '/mask/'))

    reference_frame_list = np.load('./data/frame_reference.npy')
    ref_frame_pos = 0

    for vol_path, seg_path in zip(path_volumes, path_segmentation):
        print(vol_path)
        roi_name = vol_path[:8] + 'roi'
        dc_seg = dicom.read_file(os.path.join(in_dir, 'mask', seg_path))
        seg = dc_seg.pixel_array
        seg_axial = seg[:, ::-1, :].copy()

        img_reader = monai.data.NibabelReader()
        nifti_img = img_reader.read(os.path.join(in_dir, 'CT', vol_path))
        vol = nifti_img.get_fdata()
        vol_axial = np.transpose(vol, (2, 1, 0)).copy()

        if seg_axial.shape != vol_axial.shape:
            reference_frame = reference_frame_list[ref_frame_pos]
            reference_frame_end = int(seg_axial.shape[0]) + reference_frame
            vol_axial = vol_axial[reference_frame:reference_frame_end]

            print('Reference frame:', reference_frame)
            ref_frame_pos += 1

        vol_sagital = np.transpose(vol_axial, (2, 1, 0)).copy()
        vol_coronal = np.transpose(vol_axial, (1, 2, 0)).copy()

        seg_sagital = np.transpose(seg, (2, 1, 0)).copy()
        seg_sagital = seg_sagital[:, ::-1, :]
        seg_coronal = np.transpose(seg, (1, 2, 0)).copy()
        seg_coronal = seg_coronal[::-1, :, :]

        frames_list_axial, max_frame_axial, bbx_axial = bbx_from_mask(seg_axial)
        frames_list_sagital, max_frame_sagital, bbx_sagital = bbx_from_mask(seg_sagital)
        frames_list_coronal, max_frame_coronal, bbx_coronal = bbx_from_mask(seg_coronal)

        ct_roi_axial, ct_roi_rs112_axial, ct_roi_rs64_axial = volume_from_bbx(vol_axial, frames_list_axial, bbx_axial[0])
        ct_roi_axial_3f, ct_roi_rs112_axial_3f, ct_roi_rs64_axial_3f = volume_from_bbx(vol_axial[max_frame_axial-1:max_frame_axial+2], [0, 1, 2], bbx_axial[0])
        ct_roi_sagital_3f, ct_roi_rs112_sagital_3f, ct_roi_rs64_sagital_3f = volume_from_bbx(vol_sagital[max_frame_sagital-1:max_frame_sagital+2], [0, 1, 2], bbx_sagital[0])
        ct_roi_coronal_3f, ct_roi_rs112_coronal_3f, ct_roi_rs64_coronal_3f = volume_from_bbx(vol_coronal[max_frame_coronal-1:max_frame_coronal+2], [0, 1, 2], bbx_coronal[0])


        ct_roi_9_channels_rs112 = np.concatenate((ct_roi_rs112_axial_3f, ct_roi_rs112_sagital_3f, ct_roi_rs112_coronal_3f), axis=2)
        ct_roi_9_channels_rs64 = np.concatenate((ct_roi_rs64_axial_3f, ct_roi_rs64_sagital_3f, ct_roi_rs64_coronal_3f), axis=2)

        # Full tumor
        np.save(os.path.join('data/ROIs_full/ROIs', roi_name), ct_roi_axial)
        np.save(os.path.join('data/ROIs_full/ROIs_112', roi_name), ct_roi_rs112_axial)
        np.save(os.path.join('data/ROIs_full/ROIs_64', roi_name), ct_roi_rs64_axial)

        # 3 channels
        np.save(os.path.join('data/ROIs_3frames/3channels/og_dimension', roi_name+'_axial'), ct_roi_axial_3f)
        np.save(os.path.join('data/ROIs_3frames/3channels/resized_112', roi_name+'_axial'), ct_roi_rs112_axial_3f)
        np.save(os.path.join('data/ROIs_3frames/3channels/resized_64', roi_name+'_axial'), ct_roi_rs64_axial_3f)

        # 9 channels
        np.save(os.path.join('data/ROIs_3frames/9channels/og_dimension', roi_name+'_axial'), ct_roi_axial_3f)
        np.save(os.path.join('data/ROIs_3frames/9channels/og_dimension', roi_name+'_sagital'), ct_roi_sagital_3f)
        np.save(os.path.join('data/ROIs_3frames/9channels/og_dimension', roi_name+'_coronal'), ct_roi_coronal_3f)
        np.save(os.path.join('data/ROIs_3frames/9channels/resized_112', roi_name+'_stacked'), ct_roi_9_channels_rs112)
        np.save(os.path.join('data/ROIs_3frames/9channels/resized_64', roi_name+'_stacked'), ct_roi_9_channels_rs64)

    return

in_path = '/home/ivo.navarrete/Desktop/NSCLC/data/manifest-1622561851074/NSCLC Radiogenomics'
out_path = '/home/ivo.navarrete/Desktop/NSCLC/radiogenomics/data'

# dcm2nii(in_path, out_path)
tumor_roi('/home/ivo.navarrete/Desktop/NSCLC/NSCLC_Test/data')