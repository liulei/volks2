#!/usr/bin/env python

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import time
import ctypes
import sys
import utils

def main_single():

    if len(sys.argv) < 2:
        print('./sp_calc.py .sp.npy')
        return
    name   =   sys.argv[1]
    scan_no, dm, time   =   utils.parse_name(name)
    
    cfg =   utils.gen_cfg(scan_no)    
    cfg.pols    =   [b'LL', b'RR']
    cal =   np.load('cal_fine_No%04d.npy' % (scan_no+1), allow_pickle=True).item()

    x, xe = make_sp_solve(cfg, cal, name)
    print(x)

def main_batch(scan_no):

    cfg =   utils.gen_cfg_el060(scan_no)    
    cal =   np.load('cal_fine_No%04d.npy' % (scan_no+1), allow_pickle=True).item()

    names   =   utils.gen_file_list('nsums', scan_no, 'npy')
    f   =   open('solve_No%04d.txt' % (scan_no), 'w')

    res =   []

    for name in names:
        _, dm, time  =   utils.parse_name(name)
#        print(name)
        x, xe   =   make_sp_solve(cfg, cal, name)
        if len(x) == 0:
            continue
        f.write('%11.6f\t\t%.6f\t%.6f\t%.6f\t%.6f\t%.3f\n' % \
                (time, x[0], x[1], xe[0], xe[1], dm))
        res.append(x)
    f.close()

    res =   np.array(res)
    ra  =   res[:, 0]
    dec =   res[:, 1]
    print('SP: total %d, valid %d' % (len(names), res.shape[0]))
    print('ra:  %f (%f)' % (np.mean(ra), np.std(ra)))
    print('dec: %f (%f)' % (np.mean(dec), np.std(dec)))

def make_sp_solve(cfg, cal, name):

    print(name)
    sps =   np.load(name, allow_pickle=True)

    _sps    =   utils.read_sp(name[:-4])

#    stns_new =   ['WB', 'O8', 'HH', 'Sv', 'BD', 'EF']
    stns_new =   ['BD', 'HH']

    y   =   []
    A   =   []
    sig2    =   []
    for i in range(len(sps)):
        sp  =   sps[i]
        _sp =   _sps[i]
        
#        if _sp['power'] < 4:
#            continue

        print(sp)
        if sp['snr'] < 6:
            continue

        blid    =   sp['blid']
        bl_name =   cfg.bl_no2name[cfg.bl_nos[blid]]
        stns    =   bl_name.split('-')
        if stns[0] in stns_new or stns[1] in stns_new:
            continue
        tau_cal =   cal[bl_name]['tau']
        tau_sig_cal =   cal[bl_name]['tau_sig']
        
#        y.append(_sp['mbd_match'])
        y.append(sp['tau'])
#        y.append(sp['tau'] - tau_cal)

        A.append(sp['pd'])
        sig2.append(sp['tau_sig']**2 + tau_sig_cal**2)

        if len(sys.argv) > 1:
            print('id %d, name %s, mbd %f, fit %f, snr %f' % \
                (blid, bl_name, _sp['mbd_match']*1E9, sp['tau']*1E9, sp['snr']))

    if len(A) < 3:
        return [], []

#    sig2    =   [sig ** 2] * len(y)
    
    y       =   np.array(y)[:, np.newaxis]
    A       =   np.array(A)
    Msig    =   np.diag(sig2)
    W       =   np.linalg.inv(Msig)
    At      =   A.T
    m0      =   (At @ W) @ A
    m0      =   np.linalg.inv(m0)
    x       =   ((m0 @ At) @ W) @ y

    Merr    =   m0

    sig_ra  =   np.sqrt(Merr[0, 0]) / np.pi * 180. * 60.
    sig_dec =   np.sqrt(Merr[1, 1]) / np.pi * 180. * 60.

    x   =   x / np.pi * 180. * 60.

    print('Ra: %f (%f), Dec: %f (%f)' % (x[0], sig_ra, x[1], sig_dec))
    print(x / np.pi * 180. * 60.)

    return x, np.array([sig_ra, sig_dec])

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_single()
        sys.exit(0)
#    for i in range(38, 53, 2):
    for i in range(40, 41, 2):
#    for i in range(2, 3, 1):
        main_batch(i)
