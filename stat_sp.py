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

prefix  =   './el060/nsums'

def count_bls(name):

#    stns    =   ['O8', 'BD', 'SR', 'SV', 'IR', 'ZC', 'TR', 'WB', 'IR']
    stns    =   ['O8']
#    stns    =   []
    
    with open(name, 'r') as f:
        lines   =   f.readlines()
    
    count   =   0
    for line in lines:
        if line[0] == '#':
            continue
        inc =   1
        for stn in stns:
            if stn in line:
#                print('skip %s' % (line))
                inc =   0
                continue
        count   +=  inc
    return count
        
def stat_sp(cfg, t0, t1):

    t0_sp   =   26608 + 3.162112
    p0      =   1./ (1.399541538720+-6.9885312E-3/60.)
    eps     =   0.05
    t_eps   =   p0 * eps

    def is_valid(t):

        t_sp    =   cfg.sec + t 
        dt      =   t_sp - t0_sp
        i_sp    =   int((dt + t_eps) / p0)
        if np.abs(i_sp * p0 - dt) > t_eps:
#            print('not valid: t %f, t_sp %f, n %f' % \
#                    (t, t_sp, (t_sp-t0_sp)/p0))
            return False
        return True

    n_valid =   0
    n_fake  =   0

# refer to the SP after this time
    n0  =   int((t0 + cfg.sec - t0_sp) / p0) + 1
    n1  =   int((t1 + cfg.sec - t0_sp) / p0) + 1
    
    n_pred  =   n1 - n0

    d_sel   =   {}

    names   =   utils.gen_file_list(prefix, cfg.scan_no, 'sp') 
    for name in names:
        
        _, dm, t    =   utils.parse_name(name)
        if t < t0:
            continue
        if count_bls(name) < cfg.nbl_min_crossmatch:
#            print('skip %s ... ' % (name))
            continue
        
        valid_flag  =   0
        if is_valid(t):
            valid_flag  =   1
            n_valid +=  1
        else:
            n_fake  +=  1

        d_sel[name.split('/')[-1]]  =   valid_flag

    np.save('el060/No%04d_sel.npy' % (cfg.scan_no), d_sel)

    return [n_valid, n_fake, n_pred]

def main():

    scan_nos    =   [38, 40, 42, 44, 46, 48, 50, 52]
    
    t0_vex  =   8.
    t0s =   np.array([74, 20, 20, 20, 20, 20, 20, 20]) - t0_vex
    t1s =   np.array([200, 200, 200, 200, 190, 190, 190, 190]) - t0_vex

    ns   =   []
    for i in range(len(scan_nos)):
#    for i in [0]:
        scan_no =   scan_nos[i]
        cfg =   utils.gen_cfg_el060(scan_no)
        _ns =   stat_sp(cfg, t0s[i], t1s[i])     
        ns.append(_ns)
        print(_ns)

    ns  =   np.array(ns).reshape((-1, 2, 3)).sum(axis = 1)

    ns  =   ns[::-1, :]

    srcs    =   ['OF3', 'OF2', 'OF1', 'PC'][::-1]

    t   =   (t1s - t0s).reshape((-1, 2)).sum(axis = 1)

    rate_detec  =   ns[:, 0] / ns[:, 2]
    rate_valid  =   ns[:, 0] / (ns[:, 0] + ns[:, 1])

    frac    =   [0.9, 0.5, 0.2, 0.0][::-1]

    f   =   open('stat.tex', 'w')
    for i in range(len(srcs)):
        
#        f.write('%s & %.1f & %.0f & %d & %d & %d & %.1f\\%% & %.1f\\%% \\\\\n' % \
        f.write('%s & %.1f & %.0f & %d & %d (%.1f\\%%) & %d (%.1f\\%%) \\\\\n' % \
                    (srcs[i], frac[i], t[i], \
                     ns[i, 2], ns[i, 0], rate_detec[i]*100., \
                        ns[i, 1], 100.-rate_valid[i]*100.))
    f.close()

if __name__ == '__main__':
    main()
