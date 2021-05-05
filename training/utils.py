#!/usr/bin/env python3

from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from scipy import ndimage
import numpy as np
import sys
import pickle5 as pickle


def make_data(ch=2, path="/N/project/ankur_projects/", k=14):
    '''
    Prepare data in a numpy tensor format from the pickle files.
    Args:
        ch (int): channel number (one of 0, 1, 2, 3)
        0 -> Nuclear Electrostatic Potential (NEP)
        1 -> Electron Localization Function (ELF)
        2 -> Localized Orbital Locator (LOL)
        3 -> Electrostatic Potential
        path (str): path of the directory, where data is stored
        k (int): voxel grid length of the cube 
    '''
    g_len = np.power(k, 3)
    n_chan = 4 # total number of channels in the raw data

    # Open data
    with open(path + "train_k"+str(k)+"_gr52_wfx4n.pickle", "rb") as handle:
        train_wfx = pickle.load(handle)

    with open(path + "test_k"+str(k)+"_gr52_wfx4n.pickle", "rb") as handle:
        test_wfx = pickle.load(handle)

    with open(path + "g4mp2_b3lyp_diff_labels.pickle", "rb") as handle:
        energy_diff = pickle.load(handle)

    # Load data
    X_train, y_train = load_data(energy_diff,
                                 train_wfx,
                                 dim=g_len,
                                 channels=n_chan)

    X_test, y_test = load_data(energy_diff,
                               test_wfx,
                               dim=g_len,
                               channels=n_chan)

    # Process data
    X_train_ch1, X_test_ch1 = preproc(X_train[:, :, [ch]],
                                      X_test[:, :, [ch]],
                                      scal='noscal',
                                      filt='no',
                                      sig=2.0,
                                      fun='g',
                                      ws=3,
                                      sb='no')

    #X_train = np.concatenate((X_train_ch1, X_train_ch2), axis=4)
    #X_test = np.concatenate((X_test_ch1, X_test_ch2), axis=4)
    X_train = X_train_ch1
    X_test = X_test_ch1

    X_train = np.transpose(X_train, axes=[0, 4, 1, 2, 3])
    X_test = np.transpose(X_test, axes=[0, 4, 1, 2, 3])

    return X_train, X_test, y_train, y_test


def norm_dic(channel, min_val):
    '''
    normalize dictionary values
    '''
    max_val = max(channel.values())
    fac = float(1.0 / (max_val - min_val))
    for v in channel:
        channel[v] = (channel[v] - min_val) * fac
    return channel


def gauss3D(shape=(3, 3, 3), sigma=0.5):
    """
    3D gaussian mask
    """
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-m:m + 1, -n:n + 1, -p:p + 1]
    h = np.exp(-(x * x + y * y + z * z) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def wave3D(shape=(3, 3, 3), sigma=0.5):
    """
    3D wave transform mask
    """
    omega = 1.0 / sigma
    m, n, p = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-m:m + 1, -n:n + 1, -p:p + 1]
    h = np.exp(-(x * x + y * y + z * z) / (2. * sigma * sigma)) * np.cos(
        2.0 * np.pi * omega * np.sqrt(x * x + y * y + z * z))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def app_filt(mgrid, sigma=3.0, func='g', wsize=3, sbool='no'):
    '''
    Apply Gaussian or Wave transform filter on 3D discreet data structure
    '''
    mgrid = mgrid.astype(float)
    cov = {1: 0.76, 0: 0.31, 2: 0.71, 3: 0.66, 4: 0.57}
    vdw = {1: 1.7, 0: 1.09, 2: 1.55, 3: 1.52, 4: 1.47}
    cov = norm_dic(cov, 0.0)
    vdw = norm_dic(vdw, 0.0)
    mgrid2 = []
    for i in range(m_test.shape[0]):
        temp2 = []
        for j in range(m_test.shape[4]):
            if sbool == 'no':
                scal_sigma = sigma
            else:
                #scal_sigma = sigma*vdw.get(j)
                scal_sigma = sigma * vdw.get(j + 1)
            if func == 'g':
                filt = gauss3D((wsize, wsize, wsize), scal_sigma)
            elif func == 'w':
                filt = wave3D((wsize, wsize, wsize), scal_sigma)
            else:
                print('Invalid filter function argument.', flush=True)
            temp = ndimage.filters.convolve(m_test[i, :, :, :, j],
                                            filt,
                                            mode='constant',
                                            cval=0.0)
            #temp = ndimage.filters.convolve(m_test[i,:,:,:,j], filt)
            temp2.append(temp)
        m_test2.append(temp2)

    mgrid2 = np.array(mgrid2)
    mgrid2 = np.transpose(mgrid2, (0, 2, 3, 4, 1))
    return mgrid2


def reshape_grid_arr(in_arr):
    grid_l = int(np.rint(np.cbrt(in_arr.shape[1])))
    out_arr = np.array(in_arr)
    out_arr = np.reshape(
        out_arr, (in_arr.shape[0], grid_l, grid_l, grid_l, in_arr.shape[2]))
    #out_arr = np.expand_dims(out_arr, axis=4)
    return out_arr


def preproc(train_in,
            test_in,
            scal='noscal',
            filt='no',
            sig=4.5,
            fun='g',
            ws=3,
            sb='no'):
    '''
    Scales all data using only the training set.
    Optionally convolves 3D discreet data by applying Gaussian or Wave-transform filter to reduce sparsity.
    Finally, reshapes the data into a 4D-tensor.
    Args:
        train_in (numpy array): input training data
        test_in (numpy array): input test data
        scal (str): scaling scheme of the data
                    'noscal'-> No Scaling (default)
                    'mas' -> Max Absolute Scaler
                    'ss'-> Standard Scaler
                    'ss-wmf'-> Standard Scaler(with_mean=False)
        filt (str): whether to apply smoothing/blurring or not ('yes' or 'no' (default))
        sig (float): standard deviation
        fun (str): type of smoothing filter
                  'g': Gaussian filter (default)
                  'w': Wave-transform filter
        ws (int): Window size of the filter
        sb (str): whether to scale sigma depending on van der Waal/covalent radii of the atoms or not ('yes' or 'no' (default))
                 (only applicable for atomic number channels)
                  
    '''
    if scal == 'mas':
        scaler = MaxAbsScaler()
    elif scal == 'ss-wmf':
        scaler = StandardScaler(with_mean=False)
    elif scal == 'ss':
        scaler = StandardScaler()

    if scal != 'noscal':
        mol, gr_sz, ftr = train_in.shape
        train_in = np.reshape(train_in, newshape=(-1, ftr))
        fscal = scaler.fit(train_in)
        train_in_mas = fscal.transform(train_in)
        train_in_mas = np.reshape(train_in_mas, newshape=(mol, gr_sz, ftr))

        tmol, tgr_sz, tftr = test_in.shape
        test_in = np.reshape(test_in, newshape=(-1, tftr))
        test_in_mas = fscal.transform(test_in)
        test_in_mas = np.reshape(test_in_mas, newshape=(tmol, tgr_sz, tftr))

        train_inpp = reshape_grid_arr(train_in_mas)
        test_inpp = reshape_grid_arr(test_in_mas)
    else:
        train_inpp = reshape_grid_arr(train_in)
        test_inpp = reshape_grid_arr(test_in)

    if filt == 'yes':
        train_inpp = app_filt(train_inpp,
                              sigma=sig,
                              func=fun,
                              wsize=ws,
                              sbool=sb)
        test_inpp = app_filt(test_inpp,
                             sigma=sig,
                             func=fun,
                             wsize=ws,
                             sbool=sb)

    return train_inpp, test_inpp


def load_data(energy_diff, data_wfx, dim=2197, channels=17):
    '''
    Load data into numpy arrays from dictionaries
    '''
    n_samples = len(data_wfx)
    X = np.zeros((n_samples, dim, channels))
    y = np.zeros((n_samples))

    for i, (k, v) in enumerate(data_wfx.items()):
        out = energy_diff[k + '.xyz']
        X[i, :] = v
        y[i] = out

    X = np.array(X)
    y = np.array(y)

    return X, y
