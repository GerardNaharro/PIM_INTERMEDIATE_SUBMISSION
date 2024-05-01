from matplotlib import pyplot as plt, animation
import matplotlib
import numpy as np
import pydicom
import glob
import scipy
import os

if __name__ == '__main__':
    reference_path = "part_2/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm"
    reference = pydicom.read_file(reference_path)
    img_reference = reference.pixel_array

    input_path = "part_2/RM_Brain_3D-SPGR"
    # load the DICOM files
    input_slices = []
    # Use glob to get all the files in the folder
    files = glob.glob(input_path + '/*')
    for fname in files:
        print("loading: {}".format(fname))
        input_slices.append(pydicom.dcmread(fname))
    print("CT slices loaded! \n")

    # Sort according to ImagePositionPatient header, based on the 3rd component (which is equal to "SliceLocation")
    print("Sorting slices...")
    input_slices = sorted(input_slices, key=lambda ds: float(ds.ImagePositionPatient[2]))
    print("Slices sorted!")

    # pixel aspects, assuming all slices are the same
    ps = input_slices[0].PixelSpacing
    ss = input_slices[0].SliceThickness
    ax_aspect = ps[1] / ps[0]
    sag_aspect = ps[1] / ss
    cor_aspect = ss / ps[0]

    # create 3D array
    img_shape = list(input_slices[0].pixel_array.shape)
    img_shape.append(len(input_slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(input_slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    print("shape input img3d: " + str(img3d.shape))
    print("shape reference img" + str(img_reference.shape))
