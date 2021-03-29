#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import time
import ctypes
import sys
import utils

def fit_if(cfg, buf):

    df  =   cfg.bw / cfg.nchan
    xf  =   np.arange(cfg.nchan) * df

    d_ifs   =   {}
    s_ifs  =   {}
    for fid in range(cfg.nfreq):

        arr     =   buf[fid, :]
        tau, _  =   utils.fine_fit(arr, df)
        dphs    =   2. * np.pi * xf * tau 
        
        d_ifs[fid]  =   tau
        s_ifs[fid]  =   np.sum(arr * np.exp(-1j * dphs))
        
    return d_ifs, s_ifs
             
def rot_if(cfg, buf, d_ifs, s_ifs):

    buf1    =   buf.copy()

    res =   cfg.bw / cfg.nchan
    for fid in range(cfg.nfreq):
        for ivis in range(1, cfg.nchan):
            if cfg.sbs[fid] == 'U':
                dph =   np.angle(s_ifs[fid]) + 2. * np.pi * res * ivis * d_ifs[fid]
                buf1[fid, ivis]  *=  np.exp(-1j * dph)
            else:
                dph =   np.angle(s_ifs[fid]) - 2. * np.pi * res * ivis * d_ifs[fid]
                buf1[fid, -ivis - 1] *=  np.exp(-1j * dph)
    return buf1

def gen_cal_bl(cfg, bl_no, t0, t1):

#    bl_no   =   cfg.bls[bl_id]

    heads, bufs, arr2recs   =   cfg.load_seg(bl_no, t0, t1)

    assert bufs != {}

    d   =   {}

    for i in range(len(cfg.pols)):

        pol =   cfg.pols[i]
        d[pol]  =   {}

#        print('%s:' % (cfg.pols[i]))

        buf =   bufs[cfg.pols[i]].copy()
        buf =   np.sum(buf, axis = 0) # ap, freq, chan -> freq, chan

# set DC to 0
        for fid in range(cfg.nfreq):
            if cfg.sbs[fid] == 'U':
                buf[fid, 0]     =   0.0
            else:
                buf[fid, -1]    =   0.0

        utils.plot_if(cfg, buf, 'cal_raw_%s_%s' % (cfg.bl_no2name[bl_no], pol.decode('utf-8')))

        buf1    =   buf

        d_ifs, s_ifs    =   fit_if(cfg, buf1)

        d[pol]['d_ifs']  =   d_ifs
        d[pol]['s_ifs']  =   s_ifs

        buf2    =   rot_if(cfg, buf1, d_ifs, s_ifs) 
        utils.plot_if(cfg, buf2, 'cal_fit_%s_%s' % (cfg.bl_no2name[bl_no], pol.decode('utf-8')))

    return d

def main_cal():
    
# el060:
    scan_no         =   36
    t0, t1          =   180, 210
    cfg =   utils.gen_cfg(scan_no)

#    scan_no         =   1
#    t0, t1          =   10, 40.
#    cfg =   utils.gen_cfg_aov025(scan_no)

    tu  =   cfg.ap
    t0  =   int(t0 / tu) * tu
    t1  =   int(t1 / tu) * tu

    d   =   {}
    for i in range(cfg.nbl):

        bl_no   =   cfg.bls[i]
        name    =   cfg.bl_no2name[bl_no]
        print('%s (%d) ...' % (name, bl_no))
        d[name]    =   gen_cal_bl(cfg, bl_no, t0, t1)
    
    np.save('cal_initial.npy', d)

if __name__ == '__main__':

    main_cal()
