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
from mpi4py import MPI
import utils
import torch
import cupy as cp

comm    =   MPI.COMM_WORLD
rank    =   comm.Get_rank()
psize   =   comm.Get_size()

tag_req     =   1
tag_task    =   2

dtype_fit = np.dtype([  ('imbd', 'i4'), \
                        ('isbd', 'i4'), \
                        ('mag',  'f4'), \
                        ('mag0', 'f4')])

def assign(cfg):

    if not os.path.exists('occupy'):
        os.mkdir('occupy')
    else:
        os.system('rm occupy/*')

    segs    =   np.arange(cfg.nt_seg)
    print('assign(): waiting for connection...')
    for id_seg in segs:
        rank =  comm.recv(source = MPI.ANY_SOURCE, tag = tag_req)
        print('Receive req from rank %d ...' % (rank))
        comm.send((id_seg, ), dest = rank, tag = tag_task)
        print('Seg %d, send to proc %d.' % (id_seg, rank))

    count_calc  =   0
    ncalc   =   psize - 1
    while count_calc < ncalc:
        rank =  comm.recv(source = MPI.ANY_SOURCE, tag = tag_req)
        comm.send((-1,), dest = rank, tag = tag_task) 
        count_calc  +=  1

def calc(cfg):

    while True:
        comm.send(rank, dest = 0, tag = tag_req)
        (id_seg, ) =   comm.recv(source = 0, tag = tag_task)
        print('proc %d, receive seg %d ...' % (rank, id_seg))
        if id_seg < 0:
            break
        t_id    =   id_seg
        t0      =   t_id * cfg.t_seg
        t1      =   t0 + cfg.t_seg
        if t1 > cfg.dur:
            t1  =   cfg.dur
            tsum    =   cfg.ap * cfg.nsums[-1]
            t1  =   int(t1 / tsum) * tsum

        for bl_id in range(cfg.nbl):
            id_seg  =   t_id * cfg.nbl + bl_id
            bl_no   =   cfg.bls[bl_id]
            fname   =   'No%04d/seg%04d.npy' % (cfg.scan_no, id_seg)
            if os.path.exists(fname):
                print('File %s already exists, skip.' % (fname))
                continue
            print('scan %d, %fs - %fs, bl_id %d ... ' % \
                    (cfg.scan_no, t0, t1, bl_id))
            if cfg.use_dev == 'cupy':
                ds  =   fit_cupy(cfg, t0, t1, bl_no) 
            if cfg.use_dev == 'torch':
                ds  =   fit_torch(cfg, t0, t1, bl_no) 
            if cfg.use_dev == 'numpy':
                ds  =   fit(cfg, t0, t1, bl_no) 
            
            np.save('No%04d/seg%04d.npy' % (cfg.scan_no, id_seg), ds)

def test_find_max(cfg, s):

    arr =   nsum_and_find_max(cfg, s, 1)
    return arr['isbd'], arr['imbd'], arr['mag'], arr['mag0']

    mbds    =   []
    sbds    =   []
    mags    =   []
    mag0s   =   []
    
    for i in range(s.shape[0]):
        m   =   s[i, :]
        mag =   np.abs(m)
        imc =   np.argmax(mag)
        mbds.append(imc)
        sbds.append(imc)
        mags.append(mag[imc])
        mag0s.append(mag[0])

    return mbds, sbds, mags, mag0s

def test_find_max_2d(cfg, s):

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

# s: dev, (nap1, nmc*npad)
def nsum_and_find_max_cupy(cfg, s, nsum):

    s1  =   s.reshape((-1, nsum, s.shape[1])).sum(axis = 1)

#    mag2    =   s1[:, :, 0] **2 + s1[:, :, 1] **2
    mag    =   cp.abs(s1) 

    mag_max =   cp.max(mag, 1).get() / np.sqrt(nsum)

    nap =   s1.shape[0]
    arr =   np.zeros(nap, dtype = dtype_fit)    
    arr['mag']  =   mag_max
    return arr

# s: dev, (nap1, nmc*npad)
def nsum_and_find_max_torch(cfg, s, nsum):

    s1  =   s.reshape((-1, nsum, s.shape[1])).sum(axis = 1)

#    mag2    =   s1[:, :, 0] **2 + s1[:, :, 1] **2
    mag    =   torch.abs(s1) 

    mag_max =   torch.max(mag, 1).values.cpu().numpy() / np.sqrt(nsum)

    nap =   s1.shape[0]
    arr =   np.zeros(nap, dtype = dtype_fit)    
    arr['mag']  =   mag_max
    return arr

def nsum_and_find_max(cfg, s, nsum):

#    return nsum_and_find_max_with_idx(cfg, s, nsum)

    s1  =   np.sum(np.reshape(s, (-1, nsum, s.shape[1])), axis = 1) / np.sqrt(nsum)
    mag =   np.abs(s1)

    nap =   s1.shape[0]
    arr =   np.zeros(nap, dtype = dtype_fit)    
    arr['mag']  =   np.max(mag, axis = 1)
#    arr['isbd'] =   np.argmax(mag, axis = -1)
#    arr['mag']  =   mag[np.arange(nap), arr['isbd']]
#    arr['mag0'] =   mag[np.arange(nap), 0]
#    arr['imbd'] =   arr['isbd']
    return arr

def nsum_and_find_max_with_idx(cfg, s, nsum):

    s1  =   np.sum(np.reshape(s, (-1, nsum, s.shape[1])), axis = 1) / np.sqrt(nsum)

    mag =   np.abs(s1)

    nap =   s1.shape[0]
    arr =   np.zeros(nap, dtype = dtype_fit)    
    arr['isbd'] =   np.argmax(mag, axis = -1)

    arr['mag']  =   mag[np.arange(nap), arr['isbd']]
    arr['mag0'] =   mag[np.arange(nap), 0]

    arr['imbd'] =   arr['isbd']
    return arr

def nsum_and_find_max_2d(cfg, s, nsum):

    s1  =   np.sum(np.reshape(s, (-1, nsum, s.shape[1], s.shape[2])), axis = 1) / np.sqrt(nsum)
    
    def find_max(m):
        mag   =   np.abs(m)
        imb, isb = argmax2d(mag)
        return imb, isb, mag[imb, isb], mag[s1.shape[1]//2, s1.shape[2]//2] 

    arr =   np.zeros(s1.shape[0], dtype = dtype_fit)    
    arr['imbd'], arr['isbd'], arr['mag'], arr['mag0']   =   \
            zip(*list(map(find_max, s1)))
    return arr

def open_torch(cfg):

    cfg.dev_count   =   torch.cuda.device_count()
    cfg.dev_id      =   rank % cfg.dev_count
    cfg.dev         =   torch.device('cuda:%d' % (cfg.dev_id))
#    cfg.dev_dummy   =   torch.zeros((256), dtype=torch.float, device = cfg.dev)
    print('rank %d, set to device %d' % (rank, cfg.dev_id))

def open_cupy(cfg):

    cfg.dev_count   =   cp.cuda.runtime.getDeviceCount()
    cfg.dev_id      =   rank % cfg.dev_count
    cp.cuda.Device(cfg.dev_id).use()
    print('rank %d, set to device %d' % (rank, cfg.dev_id))

# 1D fit
def fit_cupy(cfg, t0, t1, bl_no):
    
    t11     =   t1 + cfg.t_extra
    if t11 > cfg.dur:
        t11 =   cfg.dur
        t11 =   int(t11 / cfg.tsum) * cfg.tsum
#    print('proc %d, load_seg ...' % (rank))
    t_load  =   time.time()
    heads, bufs, arr2recs   =   cfg.load_seg(bl_no, t0, t11)
    t_load  =   time.time() - t_load 
        
#    print('proc %d, load_seg done!' % (rank))
    if bufs == {}:
        return {}

    t_cal   =   time.time()
    nap1 =   int((t11 - t0) / cfg.ap + 0.5)
    buf =   np.zeros((nap1, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  utils.calibrate(cfg, bufs[pol], bl_no, pol)
    buf[:, :, 0]    =   0.0 # set DC to zero
    t_cal   =   time.time() - t_cal
    nap =   int((t1 - t0) / cfg.ap + 0.5)

    print('buf h2d, size %.1f MB...' % (buf.size * 8 / 1E6))
#    buf_d   =   torch.from_numpy(buf).cuda(device = cfg.dev).reshape((nap1, cfg.nfreq, cfg.nchan))
    buf_d   =   cp.array(buf)
    print('allocate s_d...')

    ds_dms  =   {}
    t_ded   =   0.0
    t_set   =   0.0
    t_fft   =   0.0
    t_nsum  =   0.0
    for dm in cfg.dms:

        s_d     =   cp.zeros((nap1, cfg.nmc), dtype=cp.complex64)
#        print('dm %.3f' % (dm))
        t_bm    =   time.time()
        s_d[:, :]   =   0.0
        for fid in range(cfg.nfreq):
            id_mc   =   cfg.id_mcs[fid]
# no use, never uncomment
#            s_d[:, id_mc:id_mc+cfg.nchan]    =   \
#                    buf_d[:, fid, :]
            for vid in range(cfg.nchan):
                n_dm    =   cfg.tb[dm][fid, vid]

                s_d[0:nap1 - n_dm, id_mc+vid] =   \
                    buf_d[n_dm:nap1, fid, vid]

#                s_d[:, id_mc+vid]    =   \
#                    cp.roll(buf_d[:, fid, vid], -n_dm, 0)
        cp.cuda.stream.get_current_stream().synchronize()
        t_ded   +=  time.time() - t_bm
 
#        print('fft ...')
        t_bm    =   time.time()
        r_d     =   cp.fft.fft(s_d, n = cfg.nmc * cfg.npadding, axis = 1)
        cp.cuda.stream.get_current_stream().synchronize()
        t_fft   +=  (time.time() - t_bm)

#        s_d =   []
        del s_d
    
# perform nsum:
        
#        print('nsum ...')
        t_bm    =   time.time()
        ds  =   {}
        for nsum in cfg.nsums:
            ds[nsum]    =   nsum_and_find_max_cupy(cfg, r_d[0:nap, :], nsum)
            hnsum   =   nsum // 2
            if t1 + cfg.ap * hnsum <= t11:
                ds[nsum+1]  =   nsum_and_find_max_cupy(cfg, r_d[hnsum:nap+hnsum, :], nsum)
            else:
                ds[nsum+1]  =   nsum_and_find_max_cupy(cfg, r_d[hnsum:nap-hnsum, :], nsum)
        ds_dms[dm]  =   ds
        cp.cuda.stream.get_current_stream().synchronize()
        t_nsum  +=  (time.time() - t_bm)

#        r_d =   []
        del r_d

#    buf_d   =   []
    del buf_d

    if cfg.bm:
        print('')
        print('t_load:  %f' % (t_load))
        print('t_cal:   %f' % (t_cal))
        print('t_ded:   %f' % (t_ded))
        print('t_set:   %f' % (t_set))
        print('t_fft:   %f' % (t_fft))
        print('t_nsum:  %f' % (t_nsum))

        return [t_load, t_cal, t_ded, t_set, t_fft, t_nsum]

    return ds_dms

# 1D fit
def fit_torch(cfg, t0, t1, bl_no):
    
    t11     =   t1 + cfg.t_extra
    if t11 > cfg.dur:
        t11 =   cfg.dur
        t11 =   int(t11 / cfg.tsum) * cfg.tsum
#    print('proc %d, load_seg ...' % (rank))
    t_load  =   time.time()
    heads, bufs, arr2recs   =   cfg.load_seg(bl_no, t0, t11)
    t_load  =   time.time() - t_load 
        
#    print('proc %d, load_seg done!' % (rank))
    if bufs == {}:
        return {}

    t_cal   =   time.time()
    nap1 =   int((t11 - t0) / cfg.ap + 0.5)
    buf =   np.zeros((nap1, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  utils.calibrate(cfg, bufs[pol], bl_no, pol)
    buf[:, :, 0]    =   0.0 # set DC to zero
    t_cal   =   time.time() - t_cal
    nap =   int((t1 - t0) / cfg.ap + 0.5)

#    buf_1   =   torch.from_numpy(buf).cuda(device = cfg.dev).reshape((nap1, cfg.nfreq, cfg.nchan))

#    print('buf h2d, size %.1f MB...' % (buf.size * 8 / 1E6))
    buf_d   =   torch.from_numpy(buf).cuda(device = cfg.dev).reshape((nap1, cfg.nfreq, cfg.nchan))
#    print('allocate s_d...')

    ds_dms  =   {}
    t_ded   =   0.0
    t_set   =   0.0
    t_fft   =   0.0
    t_nsum  =   0.0
    for dm in cfg.dms:

        s_d     =   torch.zeros((nap1, cfg.nmc), dtype=torch.complex64, device=cfg.dev)

#        print('dm %.3f' % (dm))
        t_bm    =   time.time()
        s_d[:, :]   =   0.0
        for fid in range(cfg.nfreq):
            id_mc   =   cfg.id_mcs[fid]
# never uncomment
#            s_d[:, id_mc:id_mc+cfg.nchan]    =   \
#                    buf_d[:, fid, :]
            for vid in range(cfg.nchan):
                n_dm    =   cfg.tb[dm][fid, vid]

                s_d[0:nap1 - n_dm, id_mc+vid] =   \
                    buf_d[n_dm:nap1, fid, vid]

#                s_d[:, id_mc+vid]    =   \
#                    buf_d[:, fid, vid].roll(-n_dm, 0)
        torch.cuda.synchronize()
        t_ded   +=  time.time() - t_bm

#        print('fft ...')
        t_bm    =   time.time()
        r_d     =   torch.fft.fft(s_d, n = cfg.nmc * cfg.npadding, dim = 1)
        torch.cuda.synchronize()
        t_fft   +=  (time.time() - t_bm)

        s_d =   []
    
# perform nsum:
        
#        print('nsum ...')
        t_bm    =   time.time()
        ds  =   {}
        for nsum in cfg.nsums:
            ds[nsum]    =   nsum_and_find_max_torch(cfg, r_d[0:nap, :], nsum)
            hnsum   =   nsum // 2
            if t1 + cfg.ap * hnsum <= t11:
                ds[nsum+1]  =   nsum_and_find_max_torch(cfg, r_d[hnsum:nap+hnsum, :], nsum)
            else:
                ds[nsum+1]  =   nsum_and_find_max_torch(cfg, r_d[hnsum:nap-hnsum, :], nsum)
        ds_dms[dm]  =   ds
        torch.cuda.synchronize()
        t_nsum  +=  (time.time() - t_bm)

        r_d =   []

    buf_d   =   []

#    torch.cuda.empty_cache()
    
    if cfg.bm:
        print()
        print('t_load:  %f' % (t_load))
        print('t_cal:   %f' % (t_cal))
        print('t_ded:   %f' % (t_ded))
        print('t_set:   %f' % (t_set))
        print('t_fft:   %f' % (t_fft))
        print('t_nsum:  %f' % (t_nsum))

        return [t_load, t_cal, t_ded, t_set, t_fft, t_nsum]


    return ds_dms

# 1D fit
def fit(cfg, t0, t1, bl_no):
    
    t11     =   t1 + cfg.t_extra
    if t11 > cfg.dur:
        t11 =   cfg.dur
        t11 =   int(t11 / cfg.tsum) * cfg.tsum
#    print('proc %d, load_seg ...' % (rank))
    t_load  =   time.time()
    heads, bufs, arr2recs   =   cfg.load_seg(bl_no, t0, t11)
    t_load  =   time.time() - t_load 
        
#    print('proc %d, load_seg done!' % (rank))
    if bufs == {}:
        return {}

    t_cal   =   time.time()
    nap1 =   int((t11 - t0) / cfg.ap + 0.5)
    buf =   np.zeros((nap1, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  utils.calibrate(cfg, bufs[pol], bl_no, pol)
    buf[:, :, 0]    =   0.0 # set DC to zero
    t_cal   =   time.time() - t_cal
    nap =   int((t1 - t0) / cfg.ap + 0.5)

    ds_dms  =   {}
    t_ded   =   0.0
    t_set   =   0.0
    t_fft   =   0.0
    t_nsum  =   0.0
    _buf    =   buf.copy()
    for dm in cfg.dms:
        
        t_bm    =   time.time()
        buf =   utils.dedispersion(cfg, _buf, dm)
        t_ded    +=  (time.time() - t_bm) 

        t_bm    =   time.time()
        s   =   np.zeros((nap1, cfg.nmc), dtype = np.complex64) 
        for fid in range(cfg.nfreq):
            id_mc   =   cfg.id_mcs[fid]
            s[:, id_mc:id_mc+cfg.nchan] =   buf[:, fid, :]
        t_set   +=  (time.time() - t_bm)
    
        t_bm    =   time.time()
        r   =   np.fft.fft(s, n = cfg.nmc * cfg.npadding, axis = -1)
        t_fft   +=  (time.time() - t_bm)
    
# perform nsum:
        
        t_bm    =   time.time()
        ds  =   {}
        for nsum in cfg.nsums:
            ds[nsum]    =   nsum_and_find_max(cfg, r[0:nap, :], nsum)
            hnsum   =   nsum // 2
            if t1 + cfg.ap * hnsum <= t11:
                ds[nsum+1]  =   nsum_and_find_max(cfg, r[hnsum:nap+hnsum, :], nsum)
            else:
                ds[nsum+1]  =   nsum_and_find_max(cfg, r[hnsum:nap-hnsum, :], nsum)
        ds_dms[dm]  =   ds
        t_nsum  +=  (time.time() - t_bm)
    
    if cfg.bm:
        print('t_load:  %f' % (t_load))
        print('t_cal:   %f' % (t_cal))
        print('t_ded:   %f' % (t_ded))
        print('t_set:   %f' % (t_set))
        print('t_fft:   %f' % (t_fft))
        print('t_nsum:  %f' % (t_nsum))

        return [t_load, t_cal, t_ded, t_set, t_fft, t_nsum]

    return ds_dms

def fit_2d(cfg, t0, t1, bl_no):
    
    t11     =   t1 + cfg.t_extra
    if t11 > cfg.dur:
        t11 =   cfg.dur
        t11 =   int(t11 / cfg.tsum) * cfg.tsum
#    print('proc %d, load_seg ...' % (rank))
    heads, bufs, arr2recs   =   cfg.load_seg(bl_no, t0, t11)
#    print('proc %d, load_seg done!' % (rank))
    if bufs == {}:
        return {}

    nap1 =   int((t11 - t0) / cfg.ap + 0.5)
    buf =   np.zeros((nap1, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  utils.calibrate(cfg, bufs[pol], bl_no, pol)
    buf =   utils.dedispersion(cfg, buf, cfg.dm)

    nsb =   cfg.nchan << 4
    nmb =   cfg.nmb
    
    s   =   np.zeros((nap1, nmb, nsb), dtype = np.complex64) 
    for fid in range(cfg.nfreq):
        id_mb   =   cfg.id_mbs[fid]
        if cfg.sbs[fid] == 'U':
            s[:, id_mb, 1:cfg.nchan]    =   buf[:, fid, 1:cfg.nchan]
        else:
            s[:, id_mb, -(cfg.nchan - 1): ] =   buf[:, fid, :-1]
    
    r   =   np.fft.fft2(s, axes = (-1, -2)) # defaults to the last 2 axes
    r   =   np.fft.fftshift(r, axes = (-1, -2)) # last 2 axes
    
# perform nsum:

    nap =   int((t1 - t0) / cfg.ap + 0.5)
    ds  =   {}
    for nsum in cfg.nsums:
        ds[nsum]    =   nsum_and_find_max(cfg, r[0:nap, :, :], nsum)
        hnsum   =   nsum // 2
        if t1 + cfg.ap * hnsum <= t11:
            ds[nsum+1]  =   nsum_and_find_max(cfg, r[hnsum:nap+hnsum, :, :], nsum)
        else:
            ds[nsum+1]  =   nsum_and_find_max(cfg, r[hnsum:nap-hnsum, :, :], nsum)
    return ds

def argmax2d(m):
    s   =   np.shape(m)[1]
    i   =   np.argmax(m)
    return (i // s, i % s)

def rot_if_sbd_mbd(cfg, buf, sbd, mbd):

    buf1    =   buf.copy()

    res =   cfg.bw / cfg.nchan
    for fid in range(cfg.nfreq):
        ph_if   =   2. * np.pi * (cfg.freqs[fid] - cfg.freq_ref) * mbd
        for ivis in range(1, cfg.nchan):
            if cfg.sbs[fid] == 'U':
                dph = ph_if + 2. * np.pi * res * ivis * sbd
                buf1[fid, ivis]  *=  np.exp(-1j * dph)
            else:
                dph = ph_if - 2. * np.pi * res * ivis * sbd
                buf1[fid, -ivis - 1] *=  np.exp(-1j * dph)
    return buf1

def combine_nsum(cfg, t_beg, d):

    for nsum in cfg.nsums:
        
        a0  =   d[nsum]
        a1  =   d[nsum+1]
        a   =   np.concatenate((a0, a1), axis = 0) 

        t0  =   t_beg + (np.arange(len(a0)) * nsum + nsum / 2) * cfg.ap
        t1  =   t_beg + (np.arange(len(a0)) * nsum + nsum) * cfg.ap
        t   =   np.concatenate((t0, t1), axis = 0) 
    
        ids =   np.argsort(t)
        a   =   a[ids]
        t   =   t[ids]

        d.pop(nsum)
        d.pop(nsum+1)
        d[nsum]    =   {}
        d[nsum]['t']    =   t
        d[nsum]['p']    =   a['mag']
        d[nsum]['p0']   =   a['mag0']
        d[nsum]['mbd']  =   cfg.tau_mc[a['imbd']]
        d[nsum]['sbd']  =   cfg.tau_mc[a['isbd']]

    return d

def combine_nsum_2d(cfg, t_beg, d):

    nsb =   cfg.nchan << 4
    nmb =   cfg.nmb

    mb_res  =   1E9 / cfg.df_mb / nmb
    sb_res  =   1E9 / (cfg.bw / cfg.nchan) / nsb
    mb_range    =   mb_res * np.arange(-nmb // 2, nmb // 2, dtype = float)
    sb_range    =   sb_res * np.arange(-nsb // 2, nsb // 2, dtype = float)

    for nsum in cfg.nsums:
        
        a0  =   d[nsum]
        a1  =   d[nsum+1]
        a   =   np.concatenate((a0, a1), axis = 0) 

        t0  =   t_beg + (np.arange(len(a0)) * nsum + nsum / 2) * cfg.ap
        t1  =   t_beg + (np.arange(len(a0)) * nsum + nsum) * cfg.ap
        t   =   np.concatenate((t0, t1), axis = 0) 
    
        ids =   np.argsort(t)
        a   =   a[ids]
        t   =   t[ids]

        d.pop(nsum)
        d.pop(nsum+1)
        d[nsum]    =   {}
        d[nsum]['t']    =   t
        d[nsum]['p']    =   a['mag']
        d[nsum]['p0']   =   a['mag0']
        d[nsum]['mbd']  =   mb_range[a['imbd']]
        d[nsum]['sbd']  =   sb_range[a['isbd']]

    return d

def combine_seg(cfg):

    ds_dms  =   {}
    
    for dm in cfg.dms:
        ds  =  {}
        for bl_id in range(cfg.nbl):
            ds[bl_id]    =   {}
            for nsum in cfg.nsums:
                ds[bl_id][nsum]  =   np.zeros(cfg.nap // nsum, dtype = dtype_fit)
                ds[bl_id][nsum+1]=   np.zeros(cfg.nap // nsum, dtype = dtype_fit)
        ds_dms[dm]  =   ds
        
    print(ds.keys())
    for id_seg in range(cfg.nseg):
#        print('seg %d ...' % (id_seg))
        d   =   np.load('No%04d/seg%04d.npy' % (cfg.scan_no, id_seg), \
                    allow_pickle = True).item()
        if len(d) == 0:
            continue
         
        t_id    =   id_seg // cfg.nbl
        bl_id   =   id_seg % cfg.nbl
        
        t0      =   t_id * cfg.t_seg

        bl_no   =   cfg.bls[bl_id]
        bl_name =   cfg.bl_no2name[bl_no]
 
        for dm in cfg.dms:
            ds  =   ds_dms[dm]
            for nsum in cfg.nsums:
                print('id_seg %d, t_id %d, bl_id %d, nsum %d' % \
                    (id_seg, t_id, bl_id, nsum))
                n0      =   t_id * cfg.nap_per_seg // nsum
                size    =   len(d[dm][nsum])
                if size != cfg.nap_per_seg // nsum:
                    print('bl %d, t_id %d, nsum %d, size %d' % \
                        (bl_id, t_id, nsum, size))
                ds[bl_id][nsum][n0:n0+size]     =   d[dm][nsum][:]

                size    =   len(d[dm][nsum+1])
#               if size != cfg.nap_per_seg // nsum:
#                   print('bl %d, t_id %d, nsum %d, size %d' % \
#                        (bl_id, t_id, nsum, size))
                ds[bl_id][nsum+1][n0:n0+size]   =   d[dm][nsum+1][:]

    for dm in cfg.dms:
        for bl_id in range(cfg.nbl):
            ds_dms[dm][bl_id]   =  combine_nsum(cfg, 0, ds_dms[dm][bl_id]) 

    np.save('No%04d/fitdump.npy' % (cfg.scan_no), ds_dms)

    f   =   open('No%04d/blinfo.txt' % (cfg.scan_no), 'w')
    for i in range(cfg.nbl):
        bl_no   =   cfg.bls[i]
        f.write('%3d\t%6d\t%s\n' % (i, bl_no, cfg.bl_no2name[bl_no]))
    f.close()

def test():

#    scan_no =   37
#    cfg =   utils.gen_cfg_el060(scan_no)

    scan_no =   1
    cfg =   utils.gen_cfg_aov025(scan_no)

    cfg.tsum    =   cfg.ap # set to 1 ap
    s1  =   cfg.stn2id['SH']
    s2  =   cfg.stn2id['T6']
    bl_no   =   (s1+1) * 256 + (s2+1) 
    t0, t1  =   25., 55.
#    t0, t1  =   110, 150.
    t0  =   int(t0 / cfg.tsum) * cfg.tsum
    t1  =   int(t1 / cfg.tsum) * cfg.tsum
    t11 =   t1

    heads, bufs, arr2recs   =   cfg.load_seg(bl_no, t0, t1)
    if bufs == {}:
        return {}

    nap1 =   int((t11 - t0) / cfg.ap + 0.5)

    buf =   np.zeros((nap1, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  bufs[pol]
    utils.plot_if(cfg, np.sum(buf, axis = 0), 'raw')
 
    buf =   np.zeros((nap1, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  utils.calibrate(cfg, bufs[pol], bl_no, pol)
    utils.plot_if(cfg, np.sum(buf, axis = 0), 'cal')

    nap =   int((t1 - t0) / cfg.ap + 0.5)
    s   =   np.zeros((nap, cfg.nmc), dtype = np.complex64) 
    for fid in range(cfg.nfreq):
        id_mc   =   cfg.id_mcs[fid]
        s[:, id_mc+1:id_mc+cfg.nchan] =   buf[:nap, fid, 1:]
    
    r   =   np.fft.fft(s, n = cfg.nmc * cfg.npadding, axis = -1)
   
    r   =   np.sum(r, axis = 0)[np.newaxis, :]

    imbds, isbds, mags, mag0s =   test_find_max(cfg, r)
    print(imbds)
    print(isbds)
    print(mags)
    print(mag0s)

    nap1    =   1
    
    s   =   np.sum(s, axis = 0)
    for i in range(nap1):

        mcd =   cfg.tau_mc[imbds[i]]

        print('mcd: %.3f ns' % (mcd * 1E9))
 
        s1  =   s * np.exp(-1j * 2 * np.pi * mcd * np.arange(cfg.nmc) * cfg.bw / cfg.nchan)

        buf1    =   []
        for fid in range(cfg.nfreq):
            id_mc   =   cfg.id_mcs[fid]
            buf1.append(s1[id_mc:id_mc+cfg.nchan])
        buf1    =   np.array(buf1)
        
        utils.plot_if(cfg, buf1, 'fit_ap%d' % (i))

    return

def test_2d():

    scan_no =   37
    cfg =   utils.gen_cfg_el060(scan_no)

    cfg.tsum    =   cfg.ap # set to 1 ap
    s1  =   cfg.stn2id['IR']
#    s2  =   cfg.stn2id['SR']
    s2  =   cfg.stn2id['MC']
    bl_no   =   (s1+1) * 256 + (s2+1) 
    t0, t1  =   25., 55.
    t0  =   int(t0 / cfg.tsum) * cfg.tsum
    t1  =   int(t1 / cfg.tsum) * cfg.tsum
    t11 =   t1

    heads, bufs, arr2recs   =   cfg.load_seg(bl_no, t0, t1)
    if bufs == {}:
        return {}

    nap1 =   int((t11 - t0) / cfg.ap + 0.5)

    buf =   np.zeros((nap1, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  bufs[pol]
    utils.plot_if(cfg, np.sum(buf, axis = 0), 'raw')
 
    buf =   np.zeros((nap1, cfg.nfreq, cfg.nchan), dtype = np.complex64)
    for pol in cfg.pols:
        buf +=  utils.calibrate(cfg, bufs[pol], bl_no, pol)

    utils.plot_if(cfg, np.sum(buf, axis = 0), 'cal')

    nsb =   cfg.nchan << 4
    nmb =   cfg.nmb
    
    s   =   np.zeros((nap1, nmb, nsb), dtype = np.complex64) 
    for fid in range(cfg.nfreq):
        id_mb   =   cfg.id_mbs[fid]
        if cfg.sbs[fid] == 'U':
            s[:, id_mb, 1:cfg.nchan]    =   buf[:, fid, 1:cfg.nchan]
        else:
            s[:, id_mb, -(cfg.nchan - 1): ] =   buf[:, fid, :-1]
    
    r   =   np.fft.fft2(s, axes = (-1, -2)) # defaults to the last 2 axes
    r   =   np.fft.fftshift(r, axes = (-1, -2)) # last 2 axes
    
    nap =   int((t1 - t0) / cfg.ap + 0.5)

    r   =   np.sum(r, axis = 0)[np.newaxis, :, :]

    imbds, isbds, mags, mag0s =   test_find_max_2d(cfg, r)
    print(imbds)
    print(isbds)
    print(mags)
    print(mag0s)

    mb_res  =   1. / cfg.df_mb / nmb
    sb_res  =   1. / (cfg.bw / cfg.nchan) / nsb
    mb_range    =   mb_res * np.arange(-nmb // 2, nmb // 2, dtype = float)
    sb_range    =   sb_res * np.arange(-nsb // 2, nsb // 2, dtype = float)

    nap1    =   1
    
    for i in range(nap1):

        sbd =   sb_range[isbds[i]]
        mbd =   mb_range[imbds[i]]

        print('sbd, mbd: ', sbd, mbd)
 
        buf =   rot_if_sbd_mbd(cfg, np.sum(buf, axis = 0), sbd, mbd)
        utils.plot_if(cfg, buf, 'fit_ap%d' % (i))

    return

def main(scan_no):

#    scan_no =   int(sys.argv[1])

    if rank == 0 and not os.path.exists('No%04d' % (scan_no)):
        os.mkdir('No%04d' % (scan_no))
    dms =   np.arange(0, 1000.0, 50.0)
    cfg =   utils.gen_cfg(scan_no, dms = dms)

    cfg.rank    =   rank
    cfg.psize   =   psize
    cfg.use_dev = 'numpy'

    if psize == 1:
        combine_seg(cfg)
        return

    cfg.use_dev = 'cupy'
#    cfg.use_dev = 'torch'
#    cfg.bm  =   True
    if cfg.use_dev == 'cupy':
        open_cupy(cfg)
    if cfg.use_dev == 'torch':
        open_torch(cfg)

    comm.Barrier()
    if psize < 2:
        if rank == 0:
            print('procfit.py: at least 2 procs are required!')
        sys.exit(0)

    if rank == 0:
        assign(cfg)
    else:
        calc(cfg)
    comm.Barrier()

def compare_result(cfg, d1, d2):

    for dm in cfg.dms:
        for nsum in cfg.nsums:

            print('dm %.3f, nsum %d:' % (dm, nsum))

            m1  =   d1[dm][nsum]['mag']
            m2  =   d2[dm][nsum]['mag']
            
            id  =   np.where(np.abs(m1 - m2) / m1 > 1E-6)[0]
            print('%d out of %d with deviation > 1E-6' % \
                    (len(id), len(m1)))
#            if nsum == 2:
#                np.save('cpu_gpu_compare.npy', (m1, m2))
#            print(m1)
#            print(m2)

def run_bm_load_seg():
    
    t_segs  =   [0.5, 1.0, 2.0, 4.0, 8.0]
    arr =   []
    for t_seg in t_segs:
        _arr    =   bm_load_seg(t_seg)
        print('%f: ' % (t_seg))
        print(_arr)
        arr.append(_arr)
    arr =   np.array(arr)
    print(arr)
    np.save('benchmark/load_seg.npy', arr)

def bm_load_seg(t_seg):

    scan_no =   50
    cfg     =   utils.gen_cfg(scan_no, t_seg = t_seg)

    s1  =   cfg.stn2id['IR']
    s2  =   cfg.stn2id['MC']
    bl_no   =   (s1+1) * 256 + (s2+1) 
 
    t0  =   25.
    t0  =   int(t0 / cfg.tsum) * cfg.tsum
    t1  =   t0 + cfg.t_seg

    t11     =   t1 + cfg.t_extra
    if t11 > cfg.dur:
        t11 =   cfg.dur
        t11 =   int(t11 / cfg.tsum) * cfg.tsum

    cfg.bm_load_seg =   True
    arr     =   cfg.load_seg(bl_no, t0, t11)
    return arr

def run_benchmark():

    devs    =   ['numpy', 'torch', 'cupy']
    t_segs  =   [0.5, 1.0, 2.0, 4.0, 8.0]

    devs    =   ['torch']
#    devs    =   ['cupy']
#    devs    =   ['numpy']

    for dev in devs:
        print('dev: %s' % (dev))
        arr =   []
        for t_seg in t_segs:
            print('  t_seg: %.1f s' % (t_seg))
            arr.append(benchmark(dev, t_seg))
        arr =   np.array(arr)
#        print('rec2buf_C testing, result will not be saved!')
        np.save('benchmark/%s.npy' % (dev), arr)

def benchmark(dev, t_seg):
 
    scan_no =   50
    cfg     =   utils.gen_cfg(scan_no, t_seg = t_seg)
    cfg.bm  =   True

    s1  =   cfg.stn2id['IR']
    s2  =   cfg.stn2id['MC']
    bl_no   =   (s1+1) * 256 + (s2+1) 
 
    t0  =   25.
    t0  =   int(t0 / cfg.tsum) * cfg.tsum
    t1  =   t0 + cfg.t_seg

    if dev == 'numpy':
        def f():
            return fit(cfg, t0, t1, bl_no) 
    elif dev == 'torch':
        open_torch(cfg)
        def f():
            return fit_torch(cfg, t0, t1, bl_no) 
    elif dev == 'cupy':
        open_cupy(cfg)
        def f():
            return fit_cupy(cfg, t0, t1, bl_no) 

    arr =   []
    for i in range(11):
        res =   f()
        arr.append(res)

    arr =   np.array(arr)
    print(arr)
    arr =   np.average(arr[1:, :], axis = 0)
    print(arr)
    return arr

def run_compare():
 
    scan_no =   50
#    dms     =   np.arange(0, 1000.0, 50.0)
#    cfg     =   utils.gen_cfg_el060(scan_no, dms = dms)
    cfg     =   utils.gen_cfg(scan_no, t_seg = 2.0)
    cfg.bm  =   True
    cfg.nsums   =   [16]

    s1  =   cfg.stn2id['IR']
    s2  =   cfg.stn2id['SV']
    bl_no   =   (s1+1) * 256 + (s2+1) 
 
    t0  =   25.
    t0  =   int(t0 / cfg.tsum) * cfg.tsum
    t1  =   t0 + cfg.t_seg

    ds  =   {}
    d   =   fit(cfg, t0, t1, bl_no)

    open_torch(cfg)
    d_d   =   fit_torch(cfg, t0, t1, bl_no)

#    open_cupy(cfg)
#    d_d   =   fit_cupy(cfg, t0, t1, bl_no)

    compare_result(cfg, d, d_d)

    return
    
    if psize < 2:
        if rank == 0:
            print('procfit.py: at least 2 procs are required!')
        sys.exit(0)

    if rank == 0:
        assign(cfg)
    else:
        calc(cfg)

if __name__ == '__main__':

#    run_bm_load_seg()
#    run_benchmark()
#    run_compare()
#    test()

    scan_nos1   =   np.arange(3, 18, 2)
    scan_nos2   =   np.arange(20, 35, 2)
    scan_nos    =   np.concatenate((scan_nos1, scan_nos2))
#    for scan_no in scan_nos:
#    for scan_no in [3]:
#        main(scan_no)
