#!/usr/bin/env python

# plot cross and auto spec of all bl and stn at given time range

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
import time
import ctypes
import sys
#from mpi4py import MPI
import utils

comm    =   MPI.COMM_WORLD
rank    =   comm.Get_rank()
size    =   comm.Get_size()

tag_req     =   1
tag_task    =   2

d_dm    =   {3: 196., 50:26.833}

dtype_fit = np.dtype([  ('imbd', 'i4'), \
                        ('isbd', 'i4'), \
                        ('mag',  'f4'), \
                        ('mag0', 'f4')])

dtype_sp = np.dtype([   ('blid',    'i4'), \
                        ('apid',    'i4'), \
                        ('nap',   'i4'), \
                        ('time',    'f8'), \
                        ('w_t',     'f8'), \
                        ('mbd',     'f8'), \
                        ('snr',     'f8')])

def bl_no2name(cfg, bl_no):
    
    stn0    =   bl_no // 256 - 1
    stn1    =   bl_no % 256 - 1
    
    name    =   '%s-%s' % (cfg.stns[stn0], cfg.stns[stn1])
    return name

# f1_in, f2_in in Hz, time diff in s
def calc_tshift(f1_in, f2_in, dm):
    
    f1  =   f1_in / 1E6
    f2  =   f2_in / 1E6

    return 4.15E6 * dm * (1. / (f1 * f1) - 1. / (f2 * f2)) * 1E-3

def argmax2d(m):
    s   =   np.shape(m)[1]
    i   =   np.argmax(m)
    return (i // s, i % s)

def plot_if(cfg, buf, name):

    rc('font', size = 14) 
    
    plt.clf()

    fig =   plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(3)
    ax  =   fig.add_subplot(111)
    
    plt.subplots_adjust(left = 0.12, right = 0.9, top = 0.95, bottom = 0.15)

    df  =   cfg.bw / cfg.nchan
    x0  =   np.arange(cfg.nchan) * df

    for i in range(cfg.nfreq):
        
        x   =   cfg.freqs[i] + x0

        deg =   np.angle(buf[i, :])
        
        plt.plot(x / 1E6, deg, 'rs', ms = 5, mew = 0)

    plt.xlim(np.min(cfg.freqs) / 1E6, (cfg.freqs[-1]+cfg.bw) / 1E6)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('Band [MHz]')
    plt.ylabel('Power', color = 'r')

#    ax.twinx()
    for i in range(cfg.nfreq):
        
        x   =   cfg.freqs[i] + x0

        amp =   np.absolute(buf[i, :])
        amp =   10. * np.log10(amp)
        
#        plt.plot(x / 1E6, amp, 'b-', ms = 5, mew = 0)

#    plt.xlim((cfg.freqs[0] - cfg.bw) / 1E6, (cfg.freqs[-1] + cfg.bw) / 1E6)
#    plt.ylabel('Amplitude [dB]', color = 'b')

    plt.savefig('if_all_%s.png' % (name))

def test_find_max(cfg, s):

    mbds    =   []
    sbds    =   []
    mags    =   []
    mag0s   =   []
    
    for i in range(s.shape[0]):
        m   =   s[i, :, :]
        mag =   np.abs(m)
        imb, isb    =   argmax2d(mag)
        mbds.append(imb)
        sbds.append(isb)
        mags.append(mag[imb, isb])
        mag0s.append(mag[s.shape[1] // 2, s.shape[2] // 2])

    return mbds, sbds, mags, mag0s

def main():

    if len(sys.argv) < 4:
        print('./showspec_all.py: scan_no t0 t1')
        sys.exit(0)

#    sp_file =   sys.argv[-1]
#    scan_no, dm, time   =   utils.parse_name(sp_file)

    scan_no =   int(sys.argv[1])

    cfg =   utils.gen_cfg(scan_no)
#    cfg.dm  =   dm

    t0  =   float(sys.argv[2])
    t1  =   float(sys.argv[3])

    stns    =   []
    bl_nos =   []
#    if True:
    if False:
        print('Reading sp file %s ...' % (sp_file))
#        arr =   np.loadtxt(sp_file, dtype = dtype_sp)
        arr =   utils.read_sp(sp_file)
        for a in arr:
            blid    =   a['blid']
            s1, s2 = cfg.blid2name[blid] 
            bl_nos.append(cfg.bls[blid])
            if not s1 in stns:
                stns.append(s1)
            if not s2 in stns:
                stns.append(s2)
# Always calculate SP for all baselines
    else:
        bl_nos  =   cfg.bls
        stns    =   cfg.stns

    for i in range(cfg.nstn):
        for j in range(i+1, cfg.nstn):
            bl_no   =   (i+1)*256 + (j+1)
            if not bl_no in bl_nos:
                continue
            stn0    =   cfg.stns[i]
            stn1    =   cfg.stns[j]
            print('%s-%s...' % (stn0, stn1))
            gen_cross_spec(cfg, scan_no, t0, t1, stn0, stn1) 
            
    for stn in stns:
        print('%s-%s...' % (stn, stn))
        gen_auto_spec(cfg, scan_no, t0, t1, stn, stn) 

def gen_cross_spec(cfg, scan_no, t0, t1, stn0, stn1):

    name_png    =   'spec_No%04d_%s-%s_%.0f-%.0f.png' % (scan_no, stn0, stn1, t0, t1)
    if os.path.exists(name_png):
        return

#    stn0    =   sys.argv[4].upper()
#    stn1    =   sys.argv[5].upper()
    stn_id0 =   cfg.stn2id[stn0]
    stn_id1 =   cfg.stn2id[stn1]
    if stn_id0 > stn_id1:
        stn_id0, stn_id1 = stn_id1, stn_id0

    bl_no   =   (stn_id0 + 1) * 256 + stn_id1 + 1

    cfg.tsum    =   cfg.ap # set to 1 ap
    t0  =   int(t0 / cfg.tsum) * cfg.tsum
    t1  =   int(t1 / cfg.tsum) * cfg.tsum

    heads, bufs, arr2recs   =   cfg.load_seg(bl_no, t0, t1)
    if bufs == {}:
        return {}

    nap =   int((t1 - t0) / cfg.ap + 0.5)

    buf =   np.zeros((nap, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  utils.calibrate(cfg, bufs[pol], bl_no, pol)

#    buf =   utils.dedispersion(cfg, buf, cfg.dm)

    nsum    =   4

    nap =   int(buf.shape[0] / nsum) * nsum
    buf =   buf[:nap, :, :]

    fmin    =   cfg.freqs[0]
    fmax    =   cfg.freqs[-1] + cfg.bw
 
    df_mc   =   cfg.bw / cfg.nchan
    bw_mc   =   fmax - fmin
    nchan_mc    =   int(bw_mc / df_mc + 0.5)

    arr =   np.zeros((nap, nchan_mc), dtype=np.complex128)
    for fid in range(cfg.nfreq):
        for vid in range(1, cfg.nchan):
            f   =   cfg.freqs[fid] + vid * df_mc
            mid =   int((f-fmin) / df_mc + 0.5)
            arr[:, mid]    =   buf[:, fid, vid]

    arr =   np.reshape(arr, (-1, nsum, nchan_mc))
    arr =   np.sum(arr, axis = 1)
    nap =   int(nap / nsum)
    xf  =   np.arange(nchan_mc) * df_mc
    for i in range(nap):
        tau, _  =   utils.fine_fit(arr[i, :], df_mc) 
        arr[i, :]   *=  np.exp(-1j * 2 * np.pi * xf * tau)

    nvsum   =   16
    arr =   np.reshape(arr, (nap, -1, nvsum))
    arr =   np.sum(arr, axis = -1)
    
    extent  =   (t0, t1, fmin, fmax)
    plot_spec(arr, extent, name_png)

def norm(arr):
    sigma   =   np.std(arr)
    avg     =   np.average(arr)
    arr =   (arr - avg) / sigma
    return arr

def plot_spec(fb, extent, name):

    nsum_t  =   1
    nsum_f  =   1
    nt  =   int(fb.shape[0] / nsum_t) * nsum_t
    fb  =   fb[0:nt, :]
    fb =   np.reshape(fb, (fb.shape[0]//nsum_t, nsum_t, fb.shape[1]//nsum_f, nsum_f))
    fb  =   np.sum(fb, axis = (1, 3)) / (nsum_t * nsum_f)
    fb  =   np.absolute(fb)

    for i in range(fb.shape[1]):
        fb[:, i]    =   norm(fb[:, i])

    plt.clf()
    rc('font', size = 14)
    fig = plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(4)
    ax  =   fig.add_subplot(111)
    plt.subplots_adjust(left = 0.08, right = 0.91, top = 0.95, bottom = 0.15)
    plt.imshow(np.transpose(fb), extent = extent, aspect = 'auto', cmap = plt.get_cmap('plasma'), vmin = 0, vmax = 5, origin = 'lower')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    cax =   plt.axes([0.92, 0.15, 0.03, 0.8])
    plt.colorbar(cax = cax)
    cax.yaxis.set_ticks_position('right')
    cax.set_ylabel('Power')

    plt.savefig(name)

def gen_auto_spec(cfg, scan_no, t0, t1, stn0, stn1):

    name_png    =   'auto_No%04d_%s-%s_%.0f-%.0f.png' % (scan_no, stn0, stn1, t0, t1)
    if os.path.exists(name_png):
        return

    stn_id0 =   cfg.stn2id[stn0]
    stn_id1 =   cfg.stn2id[stn1]
    if stn_id0 > stn_id1:
        stn_id0, stn_id1 = stn_id1, stn_id0

    bl_no   =   (stn_id0 + 1) * 256 + stn_id1 + 1

    cfg.tsum    =   cfg.ap # set to 1 ap
    t0  =   int(t0 / cfg.tsum) * cfg.tsum
    t1  =   int(t1 / cfg.tsum) * cfg.tsum

    heads, bufs, arr2recs   =   cfg.load_seg(bl_no, t0, t1)
    if bufs == {}:
        return {}

    nap =   int((t1 - t0) / cfg.ap + 0.5)
    buf =   np.zeros((nap, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  bufs[pol]
 
    fmin   =   cfg.freqs[0]
    fmax   =   cfg.freqs[-1] + cfg.bw
    
    df  =   cfg.bw / cfg.nchan
    nf  =   int((fmax - fmin) / df + 0.5)

    fb  =   np.zeros((nap, nf), dtype = float)
   
    for fid in range(cfg.nfreq):
        for vid in range(cfg.nchan):
            f   =   cfg.freqs[fid] + df * vid
            if cfg.sbs[fid] ==  'L':
                f   =   cfg.freqs[fid] - cfg.bw + df * (vid + 1)
            fb_id   =   int((f - fmin)  / df + 0.5)
#            fb[:, fb_id]    =   norm(np.real(buf[:, fid, vid]))
            fb[:, fb_id]    =   np.real(buf[:, fid, vid])

    extent  =   (t0, t1, fmin, fmax)
    plot_spec(fb, extent, name_png)

if __name__ == '__main__':
    main()
