#!/usr/bin/env python3

import argparse
import sys
import os
from natsort import natsorted
import multiprocessing as mp
import numpy as np
import re
from sklearn import decomposition


def xyz2pca(args):
    '''
    Args:
        filename: xyz filename
        wfx_path: folder path where wfx files would be stored
    Output:
        Generates a new molecular orientation and writes it to a Gaussian input file
    '''
    filename, wfx_path = args
    try:
        coord_file = open(filename, 'r').read().split("\n")
    except:
        print("File ", filename, "not found!", flush=True)
        sys.exit()

    file0 = os.path.splitext(filename)[0]

    atomsymbol = []
    og_coord = []
    for line in coord_file:
        coord = re.search(
            '^\s*([A-z])\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', line)
        if coord:
            atomsymbol.append(coord.group(1))
            og_coord.append([
                np.float(coord.group(2)),
                np.float(coord.group(3)),
                np.float(coord.group(4))
            ])

    atomsymbol = np.array(atomsymbol)
    og_coord = np.array(og_coord)

    ha_coord = []
    ha_charge = []
    for id, at in enumerate(atomsymbol):
        if at != 'H':
            ha_coord.append(og_coord[id, :])

    ha_coord = np.array(ha_coord)
    ha_coord_m = np.mean(ha_coord, axis=0)
    ham_coord = og_coord - ha_coord_m
    ham_coord2 = ha_coord - ha_coord_m

    pca = decomposition.PCA(n_components=3)
    regexp = re.compile(r'00000[1-8]')
    # for molecules with only one heavy atom
    # GDB_ID: 000001, 000002, 000003, 000004, 000005, 000006,
    #         000007, 000008
    if regexp.search(file0):
        pca.fit(ham_coord)
    else:
        pca.fit(ham_coord2)   

    pca_xyz = pca.transform(ham_coord)
    pca_xyz = np.array(pca_xyz)

    pca_haxyz = []
    for id, at in enumerate(atomsymbol):
        if at != 'H':
            pca_haxyz.append(pca_xyz[id, :])
    pca_haxyz = np.array(pca_haxyz)

    xmin = np.abs(np.min(pca_haxyz[:, 0]))
    xmax = np.max(pca_haxyz[:, 0])
    xdiff = (xmax - xmin) * 0.5
    pca_xyz[:, 0] = pca_xyz[:, 0] - xdiff
    #pca_xyz[:, 0] = pca_xyz[:, 0] - 0.06347662

    at_pca_xyz = list(
        zip(atomsymbol, pca_xyz[:, 0], pca_xyz[:, 1], pca_xyz[:, 2]))

    new_gjf_name = file0 + "_pca"
    new_input = open(new_gjf_name + ".gjf", 'w')
    new_input.write('%chk=' + new_gjf_name +
                    '.chk\n%mem=3GB\n%nprocshared=2\n')
    new_input.write(
        '#p nosymm ub3lyp output=wfx pop=(npa,mk,hirshfeld) iop(6/80=1) int=grid=ultrafine 6-31g scf=xqc\n'
    )
    new_input.write('\n' + new_gjf_name + '\n\n0 1\n')
    new_input.write('\n'.join(
        ('{} {: 20.12f} {: 20.12f} {: 20.12f}'.format(*sl))
        for sl in at_pca_xyz))
    new_input.write('\n\n' + wfx_path + new_gjf_name + '.wfx\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get unique orientation of a molecule')
    parser.add_argument('--wfx_path',
                        default='/N/project/ankur_projects/',
                        type=str,
                        help='folder path of the wfx files')
    args = parser.parse_args()

    cwd = os.getcwd()
    flist = []
    for entry in os.scandir(cwd):
        if entry.is_file() and entry.name.endswith('.xyz'):
            flist.append(entry.name)

    flist = natsorted(flist)
    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)

    pool.imap_unordered(xyz2pca, [([i] + [args.wfx_path]) for i in flist])
    pool.close()
    pool.join()
