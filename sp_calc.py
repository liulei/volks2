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
import partial

libcalc =   ctypes.CDLL('calc9.1/libcalc_cwrapper.so')

def call_by_struct(din):

    dout    =   partial.COut()
    libcalc.call_calc_by_struct(ctypes.byref(din), ctypes.byref(dout))
    return np.array([dout.pd_pra, dout.pd_pdec])

def prep_din(cfg):

    din =   partial.CIn()
    partial.set_eop(din, 'el060.vex.clock')
    din.srcname =   cfg.src['name'].encode('ascii')
# in rad
    din.ra      =   cfg.src['ra']
    din.dec     =   cfg.src['dec']
    return din

# i is stn id
def calc_partial(cfg, din, i):

    s       =   cfg.s[i]
    din.x   =   s['x']
    din.y   =   s['y']
    din.z   =   s['z']
    din.axis_off    =   s['offset']
    din.stnname     =   s['name'].encode('ascii')
    return call_by_struct(din)
    
def main_single():

    if len(sys.argv) < 2:
        print('./sp_calc.py .sp.npy')
        return
    name    =   sys.argv[1]
    scan_no, dm, time   =   utils.parse_name(name)
    
    cfg =   utils.gen_cfg(scan_no)    
    din =   prep_din(cfg)

    make_fit_calc(cfg, din, name)

def main_batch(scan_no):

    cfg =   utils.gen_cfg(scan_no)    
    din =   prep_din(cfg)

    names   =   utils.gen_file_list('nsums', scan_no, 'npy')
    for name in names:
        make_fit_calc(cfg, din, name)

def make_fit_calc(cfg, din, name):

    sps =   np.load(name, allow_pickle=True)
    for sp in sps:
        mjd     =   cfg.mjd + (cfg.sec + sp['time']) / 86400.
        din.date    =   int(mjd)
        din.time    =   mjd - din.date
        
        blid    =   sp['blid']
        i0, i1  =   cfg.bl_no2stns[cfg.bl_nos[blid]]
        print(i0, i1)

        sp['pd']    =   calc_partial(cfg, din, i0) - calc_partial(cfg, din, i1)
     
    np.save(name, sps) 

if __name__ == '__main__':
#    main_single()
    for scan_no in range(40, 41, 2):
        main_batch(scan_no)
