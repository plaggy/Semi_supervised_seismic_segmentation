import numpy as np
import segyio
import os
from skimage.util import random_noise
import matplotlib.image as mpimg


"""
pad input images so the output of a model with the number of maxpooling layers
equal to the layer_depth parameter would be of the same shape as the input
"""
def pad_data(layer_depth, data):
    x_shape = data[0].shape[0]
    y_shape = data[0].shape[1]

    for i in range(layer_depth):
        x_shape = np.ceil(x_shape / 2)
        y_shape = np.ceil(y_shape / 2)

    for i in range(layer_depth):
        x_shape = int(x_shape * 2)
        y_shape = int(y_shape * 2)

    data_padded = np.pad(data, ((0, 0), (0, x_shape - data.shape[1]), (0, y_shape - data.shape[2]),
                                      (0, 0)), mode='constant', constant_values=0)

    return data_padded


# function to read in segy files
def segy_read(segy_file, mode, scale=1, inp_cube=None, read_direc='xline', inp_res=np.float32):

    if mode == 'create':
        print('Starting SEG-Y decompressor')
        output = segyio.spec()

    elif mode == 'add':
        if inp_cube is None:
            raise ValueError('if mode is add inp_cube must be provided')
        print('Starting SEG-Y adder')
        cube_shape = inp_cube.shape
        data = np.empty(cube_shape[0:-1])
    else:
        raise ValueError('mode must be create or add')

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r") as segyfile:
        segyfile.mmap()

        if mode == 'create':
            # Store some initial object attributes
            output.inl_start = segyfile.ilines[0]
            output.inl_end = segyfile.ilines[-1]
            output.inl_step = segyfile.ilines[1] - segyfile.ilines[0]

            output.xl_start = segyfile.xlines[0]
            output.xl_end = segyfile.xlines[-1]
            output.xl_step = segyfile.xlines[1] - segyfile.xlines[0]

            output.t_start = int(segyfile.samples[0])
            output.t_end = int(segyfile.samples[-1])
            output.t_step = int(segyfile.samples[1] - segyfile.samples[0])

            # Pre-allocate a numpy array that holds the SEGY-cube
            data = np.empty((segyfile.xline.length,segyfile.iline.length,\
                            (output.t_end - output.t_start)//output.t_step+1), dtype = np.float32)

        # Read the entire cube line by line in the desired direction
        if read_direc == 'inline':
            # Potentially time this to find the "fast" direction
            for il_index in range(segyfile.xline.len):
                data[il_index,:,:] = segyfile.iline[segyfile.ilines[il_index]]

        elif read_direc == 'xline':
            for xl_index in range(segyfile.iline.len):
                data[:,xl_index,:] = segyfile.xline[segyfile.xlines[xl_index]]

        elif read_direc == 'full':
            data = segyio.tools.cube(segy_file)
        else:
            print('Define reading direction(read_direc) using either ''inline'', ''xline'', or ''full''')

        factor = scale/np.amax(np.absolute(data))
        if inp_res == np.float32:
            data = (data*factor)
        else:
            data = (data*factor).astype(dtype=inp_res)

    if mode == 'create':
        output.data = data

    return output


"""
Function to read in text interpretation files in the format I used into. 
File names should have facies names in them (passed as the list in the facies_names parameter)
Returns a dictionary in the format 'slice number': array of labels with coordinates.
The array of coordinates has the following structure: [il number, xl number, class number]
"""
def convert(file_list, facies_names):
    file_list_by_facie = []
    for facie in facies_names:
        facie_list = []
        for filename in file_list:
            if facie in os.path.basename(filename):
                facie_list.append(filename)
        file_list_by_facie.append(facie_list)

    z_slices = {}

    for i, files in enumerate(file_list_by_facie):
        for filename in files:
            a = np.loadtxt(filename, skiprows=0, usecols=range(3), dtype=np.int32)
            a = np.append(a, i * np.ones((len(a), 1), dtype=np.int32), axis=1)
            if a[0][2] not in z_slices:
                z_slices[a[0][2]] = a
            else:
                z_slices[a[0][2]] = np.append(z_slices[a[0][2]], a, axis=0)

    # Return the list of adresses and classes as a numpy array
    return z_slices


"""
Function to turn the output of the 'convert' function into 2d label maps. 
"""
def point_to_images(z_points, segy_obj, slice_num=None):
    seis_data = []
    labels = []
    train_slices = []
    for z_sl, points in z_points.items():
        if slice_num is not None:
            if slice_num != z_sl:
                continue
        train_slices.append(z_sl)
        classes = np.unique(points[:, -1])
        label = np.zeros((segy_obj.data.shape[0], segy_obj.data.shape[1], len(classes) + 1))
        backgr = np.ones((segy_obj.data.shape[0], segy_obj.data.shape[1]))

        for c in classes:
            idx_c = np.where(points[:, 3] == c)[0]
            label[(points[idx_c, 0] - segy_obj.inl_start) // segy_obj.inl_step, (points[idx_c, 1] - segy_obj.xl_start) // segy_obj.xl_step, c + 1] = 1
            backgr[(points[idx_c, 0] - segy_obj.inl_start) // segy_obj.inl_step, (points[idx_c, 1] - segy_obj.xl_start) // segy_obj.xl_step] = 0

        label[..., 0] = backgr
        labels.append(label)

        seis_data.append(segy_obj.data[:, :, z_sl - segy_obj.t_start, :])

    return np.array(seis_data), np.array(labels), train_slices


"""
Extracts patches from seismic images and correcponding labels jointly.
window_size is the size of resulting patches, window_step is both the vertical and horizontal
increment between consecutive patches in the number of pixels. 
Analogous to the random cropping augmentation techniques but this one is not random

The function goes over patches locations in a loop but this can be optmized by
indexing a seismic image in the right way if needed.
"""
def patches_creator(seismic_data, labels, window_size, window_step):
    seis_patches = []
    label_patches = []

    for line, label in zip(seismic_data, labels):
        window_start_x = 0
        window_end_x = window_size
        window_start_y = 0
        window_end_y = window_size
        n_steps_x = int(np.ceil((line.shape[0] - window_size) / window_step)) + 1
        n_steps_y = int(np.ceil((line.shape[1] - window_size) / window_step)) + 1

        for i in range(n_steps_x * n_steps_y):
            classes_ex = label[window_start_x:window_end_x, window_start_y:window_end_y]
            if np.sum(classes_ex.flatten() == 1) > 0:
                label_patches.append(classes_ex)

                seis_ex = line[window_start_x:window_end_x, window_start_y:window_end_y, :]
                seis_patches.append(seis_ex)

                seis_patches.append(np.flipud(seis_ex))
                label_patches.append(np.flipud(classes_ex))

                seis_patches.append(random_noise(seis_ex, mode='gaussian', var=0.001))
                label_patches.append(classes_ex)

            if window_end_x + window_step > line.shape[0]:
                if line.shape[0] - window_end_x < window_size / 10:
                    window_start_x = 0
                    window_end_x = window_size
                    if line.shape[1] - window_end_y < window_size / 10:
                        break
                    elif line.shape[1] - window_end_y < window_size:
                        window_start_y = line.shape[1] - window_size
                        window_end_y = line.shape[1]
                    else:
                        window_start_y += window_step
                        window_end_y += window_step
                else:
                    window_start_x = line.shape[0] - window_size
                    window_end_x = line.shape[0]
            else:
                window_start_x += window_step
                window_end_x += window_step

    return np.array(seis_patches), np.array(label_patches)


"""
The main function for input data processing. Outputs the data in the appropriate format
required by the train_wrapper
"""
def load_data(segy_filename, inp_res, facies_names, train_files, slice_n=1026):
    if type(segy_filename) is str or (type(segy_filename) is list and len(segy_filename) == 1):
        # Check if the filename needs to be retrieved from a list
        if type(segy_filename) is list:
            segy_filename = segy_filename[0]

        # Make a master segy object
        segy_obj = segy_read(segy_filename, mode='create', read_direc='full', inp_res=inp_res)

        # Define how many segy-cubes we're dealing with
        segy_obj.cube_num = 1
        segy_obj.data = np.expand_dims(segy_obj.data, axis=len(segy_obj.data.shape))

    elif type(segy_filename) is list:
        # start an iterator
        i = 0

        # iterate through the list of cube names and store them in a masterobject
        for filename in segy_filename:
            # Make a master segy object
            if i == 0:
                segy_obj = segy_read(filename, mode='create', read_direc='full', inp_res=inp_res)

                # Define how many segy-cubes we're dealing with
                segy_obj.cube_num = len(segy_filename)

                # Reshape and preallocate the numpy-array for the rest of the cubes
                print('Starting restructuring to 4D arrays')
                ovr_data = np.empty((list(segy_obj.data.shape) + [len(segy_filename)]))
                ovr_data[:, :, :, i] = segy_obj.data
                segy_obj.data = ovr_data
                ovr_data = None
                print('Finished restructuring to 4D arrays')
            else:
                # Add another cube to the numpy-array
                segy_obj.data[:, :, :, i] = segy_read(segy_filename, mode='add', inp_cube=segy_obj.data,
                                                      read_direc='full', inp_res=inp_res)
            # Increase the itterator
            i += 1
    else:
        print('The input filename needs to be a string, or a list of strings')

    z_points = convert(train_files, facies_names)
    seis_data, labels, train_slices = point_to_images(z_points, segy_obj, slice_num=slice_n)

    test_seis = segy_obj.data[:, :, 1029 - segy_obj.t_start, :][30:350, 50:370, :]

    x = seis_data[:, 30:350, 50:370, 0]
    y = np.zeros((len(seis_data), 320, 320, len(facies_names) + 1))
    labels = labels[:, 30:350, 50:370]

    # this part applies a mask to turn full labels into partial labels
    # masks locations are hardcoded, can be easily changed if needed
    for i, slice_num in enumerate(train_slices):
        masks = []
        masks.append(
            mpimg.imread(os.path.dirname(segy_filename) + f'/RIPED_label_{slice_num}_no-fault_backgr_mask.jpg'))
        masks.append(
            mpimg.imread(os.path.dirname(segy_filename) + f'/RIPED_label_{slice_num}_no-fault_channel_mask.jpg'))
        mask_f = mpimg.imread(os.path.dirname(segy_filename) + f'/RIPED_label_{slice_num}_fault_mask.jpg')
        masks.append(mask_f[30:350, 50:370, :])

        masks = masks[:len(facies_names) + 1]
        for c, mask in enumerate(masks):
            idx = np.where((np.sum(mask, axis=-1) < 650) & (labels[i, ..., c] == 1))
            y[i, idx[0], idx[1], c] = labels[i, idx[0], idx[1], c]

    return x, y, labels, test_seis