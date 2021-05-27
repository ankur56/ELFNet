#!/usr/bin/env python3

import argparse
import numpy as np
import pickle5 as pickle
import sys
import os
from natsort import natsorted
import multiprocessing as mp


def proc_propfile(fname):
    data = np.genfromtxt(fname, delimiter=",")
    ind = np.lexsort((data[:, 0], data[:, 2]))
    data = data[ind]
    prop = data[:, 3]
    prop = np.expand_dims(prop, axis=1)
    return prop


def txt2dictionary(filename):
    head, sep, tail = filename.partition('_pca_k14_gr52_')
    fstr = "_pca_k14_gr52_"
    # Nuclear Electrostatic Potential (NEP)
    wfx8file = head + sep + "8.txt"
    # Electron Localization Function (ELF)
    wfx9file = head + sep + "9.txt"
    # Localized Orbital Locator (LOL)
    wfx10file = head + sep + "10.txt"
    # Electrostatic Potential (ESP)
    wfx12file = head + sep + "12.txt"

    wfx8 = proc_propfile(wfx8file)
    wfx9 = proc_propfile(wfx9file)
    wfx10 = proc_propfile(wfx10file)
    wfx12 = proc_propfile(wfx12file)
    mol_grid = np.concatenate((wfx8, wfx9, wfx10, wfx12), axis=1)
    grid_dic = {head: mol_grid}

    return grid_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'make train/test input data (pickle format) from Multiwfn property files'
    )
    parser.add_argument('--data_split',
                        default='train',
                        type=str,
                        help='use either train or test')
    args = parser.parse_args()

    cwd = os.getcwd()
    flist = []
    for entry in os.scandir(cwd):
        if entry.is_file() and entry.name.endswith('_9.txt'):
            flist.append(entry.name)

    flist = natsorted(flist)
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)

    m_grid = pool.imap_unordered(txt2dictionary, flist)
    pool.close()
    pool.join()

    m_grid_dic = {k: v for element in m_grid for k, v in element.items()}
    with open(args.data_split + "_k14_gr52_wfx4n.pickle", "wb") as handle:
        pickle.dump(m_grid_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
