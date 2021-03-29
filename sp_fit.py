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

def fit_sp(cfg, sp):

    bl_no   =   cfg.bls[sp['blid']]
    
    t0  =   cfg.ap * sp['apid']
    t1  =   t0 + sp['nap'] * cfg.ap + cfg.t_extra
    print('blid %d, %s ...' % (sp['blid'], cfg.bl_no2name[bl_no]))
    h, bufs, a2r    =   cfg.load_seg(bl_no, t0, t1)

    s0, s1  =   cfg.bl_no2stns[bl_no]

    _, as0, _    =   cfg.load_seg((s0+1)*257, t0, t1)
    _, as1, _    =   cfg.load_seg((s1+1)*257, t0, t1)

    bufs    =   utils.make_norm(bufs, as0, as1)

    _nap =   int((t1 - t0) / cfg.ap + 0.5)
    buf =   np.zeros((_nap, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  utils.calibrate(cfg, bufs[pol], bl_no, pol)
    
# load the first nap aps after dedispersion
    buf =   utils.dedispersion(cfg, buf, sp['dm'])[:sp['nap'], :, :]
    
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

    def mk_comb(arr):
        nvsum   =   128
        return np.sum(arr.reshape((-1, nvsum)), axis = 1)

    nap     =   sp['nap']
    nfreq   =   cfg.nfreq
    nchan   =   cfg.nchan
    npol    =   len(cfg.pols)
    
    snr     =   amp * np.sqrt(2 * cfg.bw * cfg.ap / (nap * nfreq * npol)) / (nchan -1)
#    snr     =   amp * np.sqrt(2 * bw_mc * cfg.ap / (nap)) / (nchan_mc - 1)
    tau_sig =   1. / (2. * np.pi * np.std(cfg.freqs) * snr)
    return arr, tau, snr, tau_sig

#    bl_name =   cfg.bl_no2name[bl_no]
#    xf  =   np.arange(nchan_mc) * df_mc
#    dphs    =   2. * np.pi * xf * tau
#    utils.plot_fine_fit(mk_comb(arr), df_mc, 'mc_raw_%s' % (bl_name))
#    arr1    =   arr * np.exp(-1j * dphs)
#    utils.plot_fine_fit(mk_comb(arr1), df_mc, 'mc_fit_%s' % (bl_name))

def prep_din():

    din =   partial.CIn()
    partial.set_eop(din, 'el060.vex.clock')
    partial.set_src(din, src)
    return din

def make_fit_sp(cfg, fname):
    sps =   utils.read_sp(fname)
    for sp in sps:
        sp['arr'], sp['tau'], sp['snr'], sp['tau_sig'] = fit_sp(cfg, sp)
#        print(sp)
    np.save('%s.npy' % (fname), sps)
    
def main_single():

    if len(sys.argv) < 2:
        print('./sp_fit.py sp_file')
        return
    fname   =   sys.argv[1]
    scan_no, dm, time   =   utils.parse_name(fname)
    cfg =   utils.gen_cfg(scan_no)
    make_fit_sp(cfg, fname)

def main():

    scan_no =   int(sys.argv[1])
    main_batch(scan_no)

def main_batch(scan_no):
    
    cfg =   utils.gen_cfg_el060(scan_no)
#    cfg =   utils.gen_cfg_aov025(scan_no)

    prefix = './nsums'    

    s1  =   'No%04d' % (scan_no)

    names   =   utils.gen_file_list(prefix, scan_no, 'sp')
    print('%d sp files in %s for scan %d' % (len(names), prefix, scan_no))
    for name in names:
        print(name)
        if os.path.exists(name+'.npy'):
            continue
        make_fit_sp(cfg, name)
  
if __name__ == '__main__':
#    main_single()
#    for scan_no in range(40, 53, 2):
#        main_batch(scan_no)
    main()
