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

def calc_effective_ap(b0):
    
    buf =   b0[:, 0, :]
    nap =   buf.shape[0]
    s   =   np.abs(np.sum(buf, axis = 1))
    nap_eff =   len(np.where(s > 1E-6)[0])
    if nap_eff < nap:
        print('nap: %d, effective: %d' % (nap, nap_eff))
    return nap_eff

def gen_fine_bl(cfg, bl_no, t0, t1):
    
    h, bufs, a2r    =   cfg.load_seg(bl_no, t0, t1)

    s0, s1  =   cfg.bl_no2stns[bl_no]

    _, as0, _    =   cfg.load_seg((s0+1)*257, t0, t1)
    _, as1, _    =   cfg.load_seg((s1+1)*257, t0, t1)

    bufs    =   utils.make_norm(bufs, as0, as1)

    nap =   int((t1 - t0) / cfg.ap + 0.5)
    buf =   np.zeros((nap, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  utils.calibrate(cfg, bufs[pol], bl_no, pol)

    nap_e   =   calc_effective_ap(buf)
    
    buf     =   np.sum(buf, axis = 0)
    fmin    =   cfg.freqs[0]
    assert fmin == np.min(cfg.freqs)
    fmax    =   cfg.freqs[-1] + cfg.bw
    bw_mc   =   fmax - fmin # m stands for multi
    df_mc   =   cfg.bw / cfg.nchan
    nchan_mc=   int(bw_mc / df_mc + 0.5)
    arr =   np.zeros(nchan_mc, dtype=np.complex128)
    mids    =   []
    for fid in range(cfg.nfreq):
        for vid in range(1, cfg.nchan):
            f   =   cfg.freqs[fid] + vid * df_mc
            mid =   int((f - fmin) / df_mc + 0.5)
            assert (not mid in mids)
            mids.append(mid)
            arr[mid]    =   buf[fid, vid]
    tau, amp    =   utils.fine_fit(arr, df_mc) 

    snr =   amp * np.sqrt(2 * cfg.bw * cfg.ap / (nap_e * cfg.nfreq * len(cfg.pols))) / (cfg.nchan - 1)

    tau_sig =   1. / (2. * np.pi * np.std(cfg.freqs) * snr)

    bl_name =   cfg.bl_no2name[bl_no]
    xf  =   np.arange(nchan_mc) * df_mc
    dphs    =   2. * np.pi * xf * tau
    utils.plot_fine_fit(arr, df_mc, 'mc_raw_No%04d_%s' % \
            (cfg.scan_no, bl_name))
    arr1    =   arr * np.exp(-1j * dphs)
    utils.plot_fine_fit(arr1, df_mc, 'mc_fit_No%04d_%s' % \
            (cfg.scan_no, bl_name))

    d   =   {}
    d['snr']    =   snr
    d['tau']    =   tau
    d['tau_sig']    =   tau_sig
    
    return d

def main(scan_no):
    
    cfg =   utils.gen_cfg(scan_no)    

    t0  =   cfg.dt(cfg.read_rec(0, count = 1)[0]) - cfg.ap/2
    t1  =   cfg.dt(cfg.read_rec(cfg.nrec - 1, count = 1)[0]) - cfg.ap/2

#    t0, t1  =   25., 55.
#    t0  =   int(t0/cfg.ap)*cfg.ap
#    t1  =   int(t1/cfg.ap)*cfg.ap
    
    d   =   {}
    for bl_no in cfg.bl_nos:
        bl_name =   cfg.bl_no2name[bl_no]
        print('ref fit %s...' % (bl_name)) 
        d[bl_name] =   gen_fine_bl(cfg, bl_no, t0, t1)

    print(d)
    np.save('cal_fine_No%04d.npy' % (scan_no), d)
    
if __name__ == '__main__':
#    scan_no =   37
    for scan_no in range(37, 54, 2):
        main(scan_no)
