#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt
import sys
import os
import time

swin_hdr = np.dtype([   ('sync', 'i4'), \
                        ('ver',  'i4'), \
                        ('no_bl','i4'), \
                        ('mjd',  'i4'), \
                        ('sec',  'f8'), \
                        ('config_idx',   'i4'), \
                        ('src_idx',      'i4'), \
                        ('freq_idx',     'i4'), \
                        ('polar',        'a2'), \
                        ('pulsarbin',    'i4'), \
                        ('weight',       'f8'), \
                        ('uvw',  'f8', 3)])

def get_occupy(cfg):
    files   =   os.listdir('./occupy')
    return len(files)

def gen_file_list(pre, no, sfx):
    
    sn  =   'No%04d' % (no)
    ss  =   os.listdir(pre) 
    names   =   []
    for s in ss:
        if s.split('_')[0] == sn and s.split('.')[-1] == sfx:
            names.append('%s/%s' % (pre, s)) 
    return names

def make_norm(bs, as0, as1):

#    return bs

    def norm(b, a0, a1):
        _a0  =   np.real(a0)
        _a1  =   np.real(a1)
        s0  =   np.average(_a0 * _a0)
        s1  =   np.average(_a1 * _a1)
        if s0 < 1E-6 or s1 < 1E-6:
#            print('make_norm(): no data is available! Skip.')
            return 0.0
        return b / np.sqrt(np.sqrt(s0 * s1))

    pols    =   bs.keys()
    for pol in pols:
        nap =   bs[pol].shape[0]
        nfreq   =   bs[pol].shape[1]
        for i in range(nap):
            for j in range(nfreq):
                bs[pol][i, j, :]  =   norm( bs[pol][i, j, :], \
                                            as0[pol][i, j, :], \
                                            as1[pol][i, j, :])
    return bs

def parse_name(name):
    
    s   =   name.split('.')
    if s[-1] == 'npy':
        s   =   name[:-7]
    elif s[-1] == 'sp':
        s   =   name[:-3]
    else:
        print('%s: unrecognized file extension!' % (name))
        return ''
# remove prefix, .sp, .sp.npy
    s   =   s.split('/')[-1].split('_')
    scan_no =   int(s[0][2:])
    if 'dm' in name:
        dm      =   float(s[1][2:])
    else:
        dm  =   0
    time    =   float(s[-1])
    
    return scan_no, dm, time

def fine_fit(arr, df):

    assert(len(arr.shape) == 1)
    nsize   =   arr.shape[0]
     
    nfft    =   1
    while nfft < nsize:
        nfft    *=  2
    nfft    *=  8

    dt      =   1. / df / nfft 
    dt      *=  1E9 # dt in ns
    d_arr   =   (np.arange(nfft) - nfft/2) * dt 
    
    r   =   np.fft.fftshift(np.fft.fft(arr, n = nfft))
    r   =   np.abs(r)
    i0   =   np.argmax(r)

# 5 points 2 order polynomial
    
    x0  =   d_arr[i0] 
    ids =   np.arange(-2, 3)
    x   =   d_arr[i0] + ids * dt
# peridoc 
    r   =   np.concatenate((r, r))
    y   =   r[i0 + ids]

#    x   =   []
#    y   =   []
#    for i in range(-2, 3): 
#        x.append(x0 + i * dt)
#        y.append(r[i0 + i])
            
    p   =   np.polyfit(x, y, 2)
    if np.abs(p[0]) < 1E-10:
        return 0.0, 0.0
    
    xmax    =   -p[1] / (2 * p[0])

#    plt.clf()
    
#    plt.plot(d_arr, r[:nfft], 'r-')     
#    plt.plot(x, np.polyval(p, x), 'c-')
#    plt.xlim(x[0] - 2*dt, x[-1]+ 2*dt)
#    plt.xlabel('Fitting delay [ns]')
#    plt.savefig('fit_plot.png')
#
#    print('p: ', p)
#    print('xmax: %f ns' % (xmax))

    return xmax/1E9, np.polyval(p, xmax)

def plot_fine_fit(arr, df, name):
    rc('font', size = 14)
    plt.clf()
    fig =   plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(3)
    ax  =   fig.add_subplot(111)
    plt.subplots_adjust(left = 0.08, right = 0.95, top = 0.95, bottom = 0.2)
    n   =   arr.shape[0]
    xf  =   np.arange(n) * df / 1E6
    deg =   np.angle(arr)
    plt.plot(xf, deg, 'rs', ms = 5, mew = 0)
    plt.ylim(-np.pi, np.pi)
    plt.xlim(xf[0], xf[-1])
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Phase [rad]')
    plt.savefig('%s.png' % (name)) 

def read_sp(name):
    
    _, dm, _    =   parse_name(name)
    sps =   []
    with open(name, 'r') as f:
        lines   =   f.readlines()
    for line in lines:
        if line.strip()[0] == '#':
            continue
        w   =   line.split()
        sp  =   {}
        sp['blid']  =   int(w[0])
        sp['apid']  =   int(w[1])
        sp['nap']   =   int(w[2])
        sp['time']  =   float(w[3])
        sp['width'] =   float(w[4])
        sp['mbd_match']   =   float(w[5]) / 1E9
        sp['power'] =   float(w[6])
        sp['dm']    =   dm

        sps.append(sp)
        
    return sps

# f1_in, f2_in in Hz, time diff in s
def calc_tshift(f1_in, f2_in, dm):
    
    f1  =   f1_in / 1E6
    f2  =   f2_in / 1E6

    return 4.15E6 * dm * (1. / (f1 * f1) - 1. / (f2 * f2)) * 1E-3

class Scan(object):
    
    def __init__(self):
        self.mjd    =   -1
        self.fullsec    =   -1
        self.dur    =   -1

class Config(object):

    def __init__(self):

        self.task   =   ''
        self.path   =   ''
        self.fmt    =   ''
        self.scan_no=   -1
        self.mjd    =   -1
        self.sec    =   -1
        self.nstn   =   -1
        self.stns   =   []
        self.nfreq  =   -1
        self.freqs  =   []
        self.sbs_raw    =   []
        self.sbs    =   []
        self.nchan  =   -1
        self.nbl    =   -1
        self.bls    =   []
        self.nseg   =   -1
        self.nt_seg =   -1
        self.nap    =   -1
        self.t_seg  =   -1
        self.ap     =   -1.0
        self.bw     =   -1.0
        self.t_u    =   -1.0
#        self.pols   =   [b'LL', b'RR', b'LR', b'RL']
        self.pols   =   [b'LL', b'RR']
        self.id_mbs =   {}
        self.df_mb  =   0.0
        self.nsums  =   [2, 4, 8, 16]
        self.dms    =   []
        self.rank   =   0
        self.psize  =   1
        self.t0     =   -1.0
        self.t1     =   -1.0

        self.sigma_winmatch     =   3.0
        self.ne_min_winmatch    =   2
        self.nbl_min_crossmatch =   5

    def load_difx_file(self, filename):
        f   =   open(filename, 'r')
        self.lines  =   f.readlines()
        f.close()

    def get_val(self, key):
        
        for line in self.lines:
            if not (key in line):
                continue
            val =   line.split(':')[1].strip()
            return val
        print('Cannot find key %s!' % (key))
        sys.exit(0)
#        return None
            
    def load_config(self, scan_no):
        
        self.scan_no    =   scan_no
        fmt =   '%s/%s.input' %  (self.path, self.fmt)
        self.load_difx_file(fmt % (scan_no))
        self.mjd    =   int(self.get_val('START MJD'))
        self.sec    =   int(self.get_val('START SECONDS'))
        self.dur    =   int(self.get_val('EXECUTE TIME')) 
        self.nbl    =   int(self.get_val('ACTIVE BASELINES'))

        self.bw     =   float(self.get_val('BW (MHZ) 0')) * 1E6 # in Hz
        self.ap     =   float(self.get_val('INT TIME'))
        self.t_u    =   self.ap
        self.nchan  =   int(self.get_val('NUM CHANNELS 0'))
        for fid in self.fids:
            self.freqs.append(float(self.get_val('FREQ (MHZ) %d' % (fid))) * 1E6) # in Hz
            self.sbs_raw.append(self.get_val('SIDEBAND %d' % (fid)))
            self.sbs.append(self.sbs_raw[-1])

            if self.flag_l2u and self.sbs[-1] == 'L':
                if self.rank == 0:
                    print('fid %d, idx %d, LSB to USB!' % (fid, len(self.sbs)-1))
                self.sbs[-1]     =   'U'
                self.freqs[-1]  -=  self.bw

        self.ids_seq_freq   =   np.argsort(self.freqs)
        self.freqs  =   np.sort(self.freqs)
        self.nfreq  =   len(self.freqs)
        assert self.nfreq == len(self.fids)

        self.nstn   =   int(self.get_val('TELESCOPE ENTRIES'))

#        self.nstn   =   4
#        self.nbl    =   6
#        print('############ el060, nbl set to 6 for testing ########')
#        self.dur    =   3
#        print('############ el060, dur set to 3 for testing ########')
 
        self.stn2id =   {}
        for i in range(self.nstn):
            self.stns.append(self.get_val('TELESCOPE NAME %d' % (i)))
            self.stn2id[self.stns[-1]]  =   i
         
        self.blid2name  =   []
        self.bl_no2stns =   {}
        self.bl_no2name =   {}
        for i in range(self.nstn):
            for j in range(i + 1, self.nstn):
                bl_no   =   (i + 1) * 256 + (j + 1)
                self.bls.append(bl_no)
#                print('bl_no: %d, %s-%s' % (bl_no, self.stns[i], self.stns[j]))
                self.blid2name.append((self.stns[i], self.stns[j]))
                self.bl_no2stns[bl_no]  =   (i, j)
                self.bl_no2name[bl_no]  =   '%s-%s' % \
                                            (self.stns[i], self.stns[j])

        self.bl_nos =   self.bls # for legacy reasons, keep bls

        if self.t_seg < 0:
            print('Please specify the duration of each seg (in sec)!')
            sys.exit(0)

        if len(self.nsums) == 0:
            print('Please set nsum before load_config!\n')
            sys.exit(0)

        t_nsum  =   self.nsums[-1] * self.ap # in ms
        if self.t_seg < t_nsum:
            self.t_seg = t_nsum
        t_seg  =   int(self.t_seg / t_nsum) * t_nsum
        if self.t_seg != t_seg:
#            print('Rounding t_seg to %f s' % (t_seg))
            self.t_seg = t_seg

        self.nap    =   int(self.dur / t_nsum) * self.nsums[-1]
        
        self.nt_seg =   int(np.ceil(self.dur / self.t_seg))
        self.nseg   =   self.nt_seg * self.nbl
        self.nap_per_seg    =   int(self.t_seg / self.ap + 0.5)

        print('scan no:     %d' % (self.scan_no))
        print('t_seg:       %f s' % (self.t_seg))
        print('scan dur:    %f s' % (self.dur))
        print('nt_seg:      %d' % (self.nt_seg))
        print('nbl:         %d' % (self.nbl))
        print('nseg:        %d' % (self.nseg))
        print('nap_per_seg: %d' % (self.nap_per_seg))

        self.swin_rec    =   np.dtype([('h', swin_hdr), ('vis', 'c8', self.nchan)])
        fmt         =   '%s/%s.difx' % (self.path, self.fmt)
        base        =   fmt % (self.scan_no)
        self.swin_name      =   '%s/DIFX_%05d_%06d.s0000.b0000' % \
                            (base, self.mjd, self.sec)
        self.rec_size   =   self.swin_rec.itemsize
        self.nrec       =   os.path.getsize(self.swin_name) // self.rec_size

# load calc file:
        fmt =   '%s/%s.calc' %  (self.path, self.fmt)
        self.load_difx_file(fmt % (scan_no)) 

        self.s  =   []
        for i in range(self.nstn):
            s   =   {}
            s['name']   =   self.get_val('TELESCOPE %d NAME' % (i))
            s['mount']  =   self.get_val('TELESCOPE %d MOUNT' % (i))
            s['offset'] =   float(self.get_val('TELESCOPE %d OFFSET (m)' % (i)))
            s['x']  =   float(self.get_val('TELESCOPE %d X (m)' % (i)))
            s['y']  =   float(self.get_val('TELESCOPE %d Y (m)' % (i)))
            s['z']  =   float(self.get_val('TELESCOPE %d Z (m)' % (i)))
            
            self.s.append(s)
        
        self.src    =   {}
        self.src['name']    =   self.get_val('SOURCE 0 NAME')
        self.src['ra']      =   float(self.get_val('SOURCE 0 RA'))
        self.src['dec']     =   float(self.get_val('SOURCE 0 DEC'))

    def get_occupy(self):
        return len(os.listdir('./occupy'))

    def read_rec(self, idx_rec, count = 1):

        huge_read   =   False
        if count > 10000:
            huge_read   =   True
        fn  =   'occupy/proc_%d' % (self.rank)
        if self.psize > 2 and huge_read:

            while True:
                time.sleep(5 * np.random.rand())
                if self.get_occupy() < 1:
                    break
#                print('proc %d, wait for read...' % (self.rank))
            
            os.system('touch %s' % (fn))

#            while True:
#                time.sleep(10 * np.random.rand())
#                if not os.path.exists(fn):
#                    os.system('touch %s' % (fn))
#                    break

#        if huge_read:
#            print('proc %d, lock and read rec ...' % (self.rank))
        rec =   np.fromfile(self.swin_name, dtype = self.swin_rec, \
                count = count, offset = idx_rec * self.rec_size)
        if huge_read:
#            print('proc %d, lock freed ...' % (self.rank))
            if self.psize > 2:
                os.system('rm %s' % (fn))
 
        return rec

    def find_t(self, t):

        def nrd(t):
            return int(t / self.ap + 0.1)

        nt    =   nrd(t)
       
        il  =   0
        ih  =   self.nrec - 1

        if nrd(self.rec_time(il)) > nt or nrd(self.rec_time(ih)) < nt:
            return -1

        while il < ih:
            im      =   (il + ih) // 2
            ntm     =  nrd(self.rec_time(im))
            if ntm == nt:
                return im 
            elif ntm < nt:
                il  =   im 
            else:
                ih  =   im
#            print('im: %d, il %d, ih %d, ntm %d' % (im, il, ih, ntm))

        return -1

    def find_t_Sep(self, t):

        def rd(t):
            return int(t / self.t_u) * self.t_u
       
        il  =   0
        ih  =   self.nrec - 1

        if self.rec_time(il) > t or self.rec_time(ih) < t:
            return -1

        trd  =   rd(t)

        while il < ih:
            im =   (il + ih) // 2
            tim     =   rd(self.rec_time(im))
            if tim == trd:
                return im 
            if tim < trd:
                il  =   im 
            else:
                ih  =   im
#            print('im: %d, il %d, ih %d, tim: %f, trd: %f' % (im, il, ih, tim, trd))

        return -1

    def get_rec_range(self, t0, t1):

#        print(t0, t1, self.ap)
        assert (t0 + 1E-8) % (self.ap) < 1E-6
        assert (t1 + 1E-8) % (self.ap) < 1E-6

        t0_f    =   self.dt(self.read_rec(0, count = 1)[0])
        t1_f    =   self.dt(self.read_rec(self.nrec - 1, count = 1)[0])
        if t0 <= t0_f and t1 >= t1_f:
            return (0, self.nrec)
        
        i0h  =   self.find_t(t0)
        i1h  =   self.find_t(t1)
        
# time range is not overlapped with swin file
        if i0h == -1 and i1h == -1:
            return (-1, -1)
            
        i0  =  i0h
        if i0h != -1:
            i0l     =   self.find_t(t0 - self.ap)
            if i0l < 0:
                i0l  =   0
            recs    =   self.read_rec(i0l, count = i0h - i0l + 1)
            for i0 in range(i0l, i0h + 1):
                if self.dt(recs[i0 - i0l]) >= t0:
                    break

        i1  =   i1h
        if i1h != -1:
            i1l     =   self.find_t(t1 - self.ap) 
            if i1l < 0:
                i1l =   0
            recs    =   self.read_rec(i1l, count = i1h - i1l + 1)
            for i1 in range(i1h, i1l - 1, -1):
                if self.dt(recs[i1 - i1l]) <= t1:
#                    print('ih: break at %d' % (i1))
                    break

        if  i0 != -1 and i1 != -1:
            return (i0, i1 + 1)
        elif i0 == -1 and i1 != -1:
            return (0, i1 + 1)
        else:
            return (i0, self.nrec)
 
    def get_rec_range_Sep(self, t0, t1):

#        assert t0 % (32 * self.ap) < 1E-8
#        assert t1 % (32 * self.ap) < 1E-8

        assert t0 % (self.ap) < 1E-8
#        assert t1 % (32 * self.ap) < 1E-8

        t0_f    =   self.dt(self.read_rec(0, count = 1)[0])
        t1_f    =   self.dt(self.read_rec(self.nrec - 1, count = 1)[0])
        if t0 <= t0_f and t1 >= t1_f:
            return (0, self.nrec)
        
        i0l  =   self.find_t(t0 - self.t_u)
        i1h  =   self.find_t(t1 + self.t_u)
        
# time range is not overlapped with swin file
        if i0l == -1 and i1h == -1:
            return (-1, -1)
            
        i0  =  i0l 
        if i0l != -1:
            i0h     =   self.find_t(t0 + self.t_u)
            recs    =   self.read_rec(i0l, count = i0h - i0l + 1)
            for i0 in range(i0l, i0h + 1):
                if self.dt(recs[i0 - i0l]) >= t0:
                    break

        i1  =   i1h
        if i1h != -1:
            i1l     =   self.find_t(t1 - self.t_u) 
            recs    =   self.read_rec(i1l, count = i1h - i1l + 1)
            for i1 in range(i1h, i1l - 1, -1):
                if self.dt(recs[i1 - i1l]) <= t1:
#                    print('ih: break at %d' % (i1))
                    break

        if  i0 != -1 and i1 != -1:
            return (i0, i1 + 1)
        elif i0 == -1 and i1 != -1:
            return (0, i1 + 1)
        else:
            return (i0, self.nrec)
            
    def rec_time(self, i):

        rec =    np.fromfile(self.swin_name, dtype = self.swin_rec, \
                count = 1, offset = i * self.rec_size)[0]
        return self.dt(rec)

    def dt(self, rec):
        return (rec['h']['mjd'] - self.mjd) * 86400. \
                + (rec['h']['sec'] - self.sec)

    def load_seg(self, bl_no, t0, t1):

        if self.t0 <= t0 and self.t1 >= t1:
            recs    =   self.recs
        else:
            self.t0 =   t0
            self.t1 =   t1
            i0, i1  =   self.get_rec_range(t0, t1)

            if i0 < 0:
                recs   =   {}
            else:
                count   =   i1 - i0
#                print('proc %d, read %d recs, size %.1f GB ... ' % \
#                    (self.rank, count, count * self.rec_size / 1E9))
                recs    =   self.read_rec(i0, i1 - i0)
#                print('proc %d, read rec done!' % (self.rank))

            self.recs   =   recs

#        if recs == {}:
# empty recs dict
        if len(recs) == 0:
            return {}, {}, {}
        
        nap     =   int((t1 - t0) / self.ap + 0.5)
        bufs    =   {}
        heads   =   {}
        arr2recs=   {}
        for pol in self.pols:
            bufs[pol]   =   np.zeros((nap, self.nfreq, self.nchan), \
                                dtype = np.complex64)
#            heads[pol]  =   np.zeros((nap, self.nfreq), dtype = swin_hdr)
#            arr2recs[pol]   =   {}

#        print('proc %d, %s recs will be loaded...' % (self.rank, len(recs)))

        nc =   0
        for i, rec in enumerate(recs):

            if rec['h']['no_bl']    !=  bl_no:
                continue
            pol =   rec['h']['polar']
            if pol not in self.pols:
                continue
            fid     =   rec['h']['freq_idx']
            if fid not in self.fids:
                continue
            t_rec   =   self.dt(rec)
            if t_rec < t0:
                continue
            apid    =   int((t_rec - t0) / self.ap)
            if apid >= nap:
                break
            idx    =   self.fid2idx[fid]
            bufs[pol][apid, idx, :]   =   rec['vis'][:]
#            heads[pol][apid, idx] =   rec['h']
#            arr2recs[pol][apid * self.nfreq + idx] =   i
            nc +=  1
#            if nc % 10000 == 0:
#                print('%d APs have been loaded.' % (nc))

#        print('proc %d, loading rec done!' % (self.rank))

        if self.flag_l2u:
            bufs    =   self.bufs_l2u(bufs)
        return heads, bufs, arr2recs

# make sure that freqs are always from low to high sequantially
    def bufs_l2u(self, bufs):

        for fid in range(self.nfreq):

            if self.sbs_raw[fid] == 'U':
                continue

            for pol in self.pols:
                buf_0   =   bufs[pol][:, fid, -1].copy()
                bufs[pol][:, fid, 1:]   =   bufs[pol][:, fid, :-1]
                bufs[pol][:, fid, 0]    =   buf_0[:]

        for pol in self.pols:
            bufs[pol][:, :, :]  =   bufs[pol][:, self.ids_seq_freq, :]

        return bufs
    
    def s2n(self, s1, s2):

        id1 =   self.stn2id[s1]
        id2 =   self.stn2id[s2]

        if id1 > id2:
            id1, id2    =   id2, id1

        return (id1 + 1) * 256 + (id2 + 1)

    def prep_dedispersion(self):

        tbs    =   {}
        for dm in self.dms:
            tb  =   np.zeros((self.nfreq, self.nchan), dtype = int)

            file   =   open('dm%.3f_shift.txt' % (dm), 'w')
            df  =   self.bw / self.nchan
            for fid in range(self.nfreq):
                for vid in range(self.nchan):
                    f       =   self.freqs[fid] + df * vid
                    if self.sbs[fid] ==  'L':
                        f   =   self.freqs[fid] - self.bw + df * (vid + 1)
                    tshift  =   calc_tshift(f, self.freq_dm, dm)
#                   print(fid, vid, f / 1E6, self.freq_dm / 1E6, tshift)
                    assert(tshift >= 0.0)
                    ishift  =   int((tshift / self.ap + 0.5))
                    tb[fid, vid] =   ishift
                    file.write('%d\t%d\t%d\n' % (fid, vid, ishift))
#                   print ('fid %d, vid %d, shift %d' % (fid, vid, ishift))
            file.close()
            tbs[dm]   =   tb
        return tbs

def calibrate(cfg, buf, bl_no, pol):
    
    name_bl =   cfg.bl_no2name[bl_no]
    cfg.cal =   np.load('cal_initial.npy', allow_pickle = True).item()
    
#    cfg.cal =   np.load('cal_%s.npy' % (cfg.task), allow_pickle = True).item()
# add EF baselines
#    d   =   np.load('cal_%s_EF.npy' % (cfg.task), allow_pickle = True).item()
#    for k, v in d.items():
#        cfg.cal[k]  =   v

    s_ifs   =   cfg.cal[name_bl][pol]['s_ifs']
    d_ifs   =   cfg.cal[name_bl][pol]['d_ifs']

    df  =   cfg.bw / cfg.nchan
    for i in range(cfg.nfreq):
        
        if cfg.sbs[i] == 'U':
            freq    =   np.arange(cfg.nchan) * df
        else:
            freq    =   (np.arange(cfg.nchan) - cfg.nchan + 1) * df

        phase   =   2. * np.pi * freq * d_ifs[i] + np.angle(s_ifs[i])
        rot     =   np.exp(-1j * phase) 

        buf[:, i, :]    *=  rot
    
    return buf

def dedispersion(cfg, buf, dm):

#    tb  =   np.zeros((cfg.nfreq, cfg.nchan), dtype = int)

# larger than cfg.nap
    nap1 =   buf.shape[0]
   
    buf1    =   np.zeros((nap1, cfg.nfreq, cfg.nchan), dtype = np.complex64)

    for fid in range(cfg.nfreq):
        for vid in range(cfg.nchan):
            n_dm    =   cfg.tb[dm][fid, vid]
            buf1[0: nap1 - n_dm, fid, vid] =   buf[n_dm: nap1, fid, vid]
    return buf1

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
        
        if cfg.sbs[i] == 'U':
            x   =   cfg.freqs[i] + x0
        else:
            x   =   cfg.freqs[i] - cfg.bw + df + x0

        deg =   np.angle(buf[i, :])
        
        plt.plot(x / 1E6, deg, 'rs', ms = 5, mew = 0)

    plt.xlim((cfg.freqs[0]) / 1E6, (cfg.freqs[-1] + cfg.bw) / 1E6)
#    plt.xlim((cfg.freqs[0] - cfg.bw) / 1E6, (cfg.freqs[0] + cfg.bw) / 1E6)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel('Band [MHz]')
    plt.ylabel('Phase [Rad]', color = 'r')

#    ax.twinx()
    for i in range(cfg.nfreq):
        
        if cfg.sbs[i] == 'U':
            x   =   cfg.freqs[i] + x0
        else:
            x   =   cfg.freqs[i] - cfg.bw + df + x0

        amp =   np.absolute(buf[i, :])
        amp =   20. * np.log10(amp)
        
#    plt.ylabel('Amplitude [dB]', color = 'b')

    plt.savefig('if_all_%s.png' % (name))

def gen_cfg_aov025(scan_no, **kw):
    cfg             =   Config()
    cfg.task        =   'aov025'
    cfg.path        =   '/data/corr/aov025'
    cfg.fmt         =   'calc_%0d'
    cfg.t_seg       =   0.5 # 30 sec per seg
    cfg.t_u         =   0.1 # 0.1 s for rounding
    cfg.flag_l2u    =   True
    cfg.fids        =   np.arange(10, 16)
    cfg.fid2idx     =   {}
    for fid in cfg.fids:
        cfg.fid2idx[fid]    =   fid - 10
    cfg.load_config(scan_no)
    cfg.pols    =   [b'RR']

# 2-D fitting
    cfg.df_mb       =   8E6
    cfg.nmb         =   64 # for fitting

# 1-D fitting, new
    cfg.df_mc       =   cfg.bw / cfg.nchan
    cfg.nmc         =   1
    _nmc            =   int((cfg.freqs[-1] + cfg.bw - cfg.freqs[0]) / cfg.df_mc + 0.5)
    while cfg.nmc < _nmc:
        cfg.nmc <<= 1
    cfg.id_mcs      =   {}

    cfg.npadding    =   4
    nfft    =   cfg.nmc * cfg.npadding
    dtau    =   1. / cfg.df_mc / nfft
    cfg.tau_mc      =   np.fft.fftshift((np.arange(nfft) - nfft/2) * dtau)

    cfg.freq_ref    =   cfg.freqs[0]

    for fid in range(cfg.nfreq):
        cfg.id_mbs[fid] =   int((cfg.freqs[fid] - cfg.freq_ref) / cfg.df_mb + 0.5)
        cfg.id_mcs[fid] =   int((cfg.freqs[fid] - cfg.freq_ref) / cfg.df_mc + 0.5)

    cfg.tsum    =   cfg.nsums[-1] * cfg.ap
    cfg.dms     =      [192.4] # J1854+0306
    cfg.freq_dm     =   cfg.freqs[-1] + cfg.bw
    t_shift         =   calc_tshift(cfg.freqs[0] - cfg.bw, cfg.freqs[-1] + cfg.bw, cfg.dms[-1])
    cfg.t_extra     =   int(np.ceil(t_shift / cfg.tsum)) * cfg.tsum
    print('t_extra:     %f s' % (cfg.t_extra))

#    cfg.tb  =   cfg.prep_dedispersion()
    cfg.tb  =   cfg.prep_dedispersion()

    return cfg

def gen_cfg_el060(scan_no, **kw):
    cfg             =   Config()
    cfg.task        =   'el060'
    cfg.path        =   '/data/corr/el060_rrat'
    if scan_no >= 36:
        cfg.path        =   '/data/corr/el060_psr'
    cfg.fmt         =   'calc_%02d'
#    cfg.t_seg       =   0.5 # 30 sec per seg
    cfg.t_seg       =   8 # 30 sec per seg
    if 't_seg' in kw.keys():
        cfg.t_seg   =   kw['t_seg']
#    print('Change t_seg to %f for testing...' % (cfg.t_seg))
    cfg.t_u         =   0.1 # 0.1 s for rounding
    cfg.flag_l2u    =   True

#    cfg.fids        =   np.arange(8)
    cfg.fids    =   np.arange(1, 8) # skip FREQ 0
    cfg.fid2idx =   {}
    for i in range(len(cfg.fids)):
        fid =   cfg.fids[i]
        cfg.fid2idx[fid]    =   i

    cfg.load_config(scan_no)

# 2-D fitting
    cfg.df_mb       =   16E6
    cfg.nmb         =   32 # for fitting

# 1-D fitting, new
    cfg.df_mc       =   cfg.bw / cfg.nchan
    cfg.nmc         =   1
    _nmc            =   int((cfg.freqs[-1] + cfg.bw - cfg.freqs[0]) / cfg.df_mc + 0.5)
    while cfg.nmc < _nmc:
        cfg.nmc <<= 1
    cfg.id_mcs      =   {}

    cfg.npadding    =   2
    nfft    =   cfg.nmc * cfg.npadding
    dtau    =   1. / cfg.df_mc / nfft
    cfg.tau_mc      =   np.fft.fftshift((np.arange(nfft) - nfft/2) * dtau)

    cfg.freq_ref    =   cfg.freqs[0]

    for fid in range(cfg.nfreq):
        cfg.id_mbs[fid] =   int((cfg.freqs[fid] - cfg.freq_ref) / cfg.df_mb + 0.5)
        cfg.id_mcs[fid] =   int((cfg.freqs[fid] - cfg.freq_ref) / cfg.df_mc + 0.5)

    cfg.tsum    =   cfg.nsums[-1] * cfg.ap
    if scan_no <= 17:
#        cfg.dms =   [0.0, 100.0, 196.] # J1819-1458
        cfg.dms =   [196.] # J1819-1458
    elif scan_no  <= 34:
#        cfg.dms  =   [0.0, 100.0, 192.4] # J1854+0306
        cfg.dms  =   [192.4] # J1854+0306
    else:
        cfg.dms  =   [26.833] # J0332+5434

    if 'dms' in kw.keys():
        cfg.dms =   kw['dms']

    cfg.freq_dm     =   cfg.freqs[-1] + cfg.bw
    t_shift         =   calc_tshift(cfg.freqs[0] - cfg.bw, cfg.freqs[-1] + cfg.bw, np.max(cfg.dms))
    cfg.t_extra     =   int(np.ceil(t_shift / cfg.tsum)) * cfg.tsum
    print('t_extra:     %f s' % (cfg.t_extra))

    cfg.tb  =   cfg.prep_dedispersion()

    return cfg

def gen_cfg(scan_no, **kw):
    
    return gen_cfg_el060(scan_no, **kw)
#    return gen_cfg_aov025(scan_no)

if __name__ == '__main__':
    main()
