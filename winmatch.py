#!/usr/bin/env python

import sys, os
from collections import OrderedDict
import numpy as np
import utils
#import matplotlib
#import matplotlib.pyplot as plt

#ambig   =   1.0E9 / 16.0E6 # in ns, psrf02
ambig   =   1.0E9 / 16.0E6 # in ns, el060
hambig  =   ambig * 0.5

#ph0_dict    =   {1: 0, 69: 0.581375, 71: 0.414097, 73: 0.246818}

def calc_psr_phase(t):

#    return 0.0

    c1  =   -6.89262836039208383e-03
    F0  =   1.399538059693
#    ph0 =   0.581375 # scan 69
#    ph0 =   0.414097 # scan 0071
#    ph0 =   0.246818 # scan 0073
#    ph0 =   ph0_dict[scan_no]
    ph0 =   0.246818 # Scan 73
    return (t - 10.000512) * (F0 + c1 / 60.) + ph0

def is_overlap(e1, e2):
#    if np.abs(e1.time - e2.time) <= np.max([e1.hrange, e2.hrange]):
    if np.abs(e1.time - e2.time) <= e1.hrange + e2.hrange:
        return True
    return False

def merge_mbd(t1, t2):
    if np.abs(t1 - t2) < ambig * 0.5:
        return (t1 + t2) * 0.5
    if t2 < t1:
        t2  +=  ambig
    else:
        t2  -=  ambig
    return (t1 + t2) * 0.5

def keep_mbd_ambig(mbd, a, hr):

    return True
    
    if mbd < a - hambig:
        mbd +=  ambig
    elif mbd > a + hambig:
        mbd -=  ambig
    if np.abs(mbd - a) < hr:
        return True
    return False


def keep_mbd(mbd, a, s):
    
    r   =   s * 2.0
    a0  =   a - r
    a1  =   a + r
    if mbd < a0:
        mbd +=  ambig
    elif mbd > a1:
        mbd -=  ambig
    if a0 <= mbd and mbd <= a1:
        return True
    return False

def calc_avg_std_mbd(mbd_l):
    
    a   =   np.average(mbd_l)
    r   =   np.std(mbd_l) * 2.0
    a0  =   a - r
    a1  =   a + r

    for i in range(len(mbd_l)):
        if mbd_l[i] < a0:
            mbd_l[i]    +=  ambig
        elif mbd_l[i] > a1:
            mbd_l[i]    -=  ambig
    return np.average(mbd_l), np.std(mbd_l)

def select_by_mbd(mbdarr, nsumlst):
    
    mbdavg  =   np.average(mbdarr)
    mbdstd  =   np.std(mbdarr)
    mbdrange    =   mbdstd * 2.0
    mbd0    =   mbdavg - mbdrange
    mbd1    =   mbdavg + mbdrange

    newarr  =   []
    nsumlst_sel =   []
    for i in range(len(mbdarr)):
        mbd     =   mbdarr[i]
        nsum    =   nsumlst[i]
        if mbd < mbd0:
            mbd +=  ambig
        elif mbd > mbd1:
            mbd -=  ambig
        if mbd >= mbd0 and mbd <= mbd1:
            newarr.append(mbd)  
            nsumlst_sel.append(nsum)
    return np.average(newarr) , np.std(newarr), nsumlst_sel

class Event(object):
    def __init__(self):
        self.time   =   0.0
        self.hrange =   0.0
        self.nsum   =   0
        self.p      =   0.0
        self.sbd    =   0.0
        self.mbd    =   0.0
        self.p0     =   0.0
        self.phase  =   0.0

    def __str__(self):
        return 't %f hrange %f nsum %d mbd %.3f sbd %.3f pwr %.3f pwr0 %.3f ph %.3f' % \
            (self.time, self.hrange, self.nsum, self.mbd, self.sbd, self.p, self.p0, calc_psr_phase(self.time))

class Candidate(object):
    def __init__(self, e):
        self.time   =   e.time
        self.hrange =   e.hrange
        self.d      =   OrderedDict()
        self.d[e.nsum]  =   [e]


    def sum_to_file_max_power(self, f):

# loop each nsum to find out pmax and collect mbd
#        p_d     =   OrderedDict()

#        mbd_l   =   []
        pmax    =   -1.0
        for nsum, e_l in self.d.items():
            p_l =   []
            for e in e_l:
                if pmax < e.p:
                    pmax    =   e.p
                    e_max   =   e
#                mbd_l.append(e.mbd)

#        mbd_avg, mbd_std    =   calc_avg_std_mbd(mbd_l)
       
        f.write("%.6f\t%.6f\t%.6f\t%12.3f\t%12.3f\t%.1f\t%.1f\t%.1f\t%d\n" % \
                (e_max.time, e_max.hrange, self.hrange, e_max.mbd*1E9, \
                 e_max.sbd*1E9, e_max.p, e_max.p0, 0.0, len(self.d)))
 
    def sum_to_file_max_width(self, f):

# loop each nsum to find out pmax and collect mbd
#        p_d     =   OrderedDict()
        mbd_l   =   []
        pmax    =   -1.0
        nsum_pmax=   -1
        for nsum, e_l in self.d.items():
            p_l =   []
            for e in e_l:
                p_l.append(e.p)
                mbd_l.append(e.mbd)
            p_avg   =   np.average(p_l)
            if pmax < p_avg:
                pmax        =   p_avg
                nsum_pmax    =   nsum
#            p_d[nsum]   =   np.average(p_l)

        mbd_avg, mbd_std    =   calc_avg_std_mbd(mbd_l)
       
        l0  =   []
        l1  =   []
        p_l =   []
        for e in self.d[nsum_pmax]:
            l0.append(e.time - e.hrange)
            l1.append(e.time + e.hrange)
            p_l.append(e.p)
        t0  =   np.min(l0)
        t1  =   np.max(l1)
        t   =   (t0 + t1) * 0.5
        hr  =   (t1 - t0) * 0.5
    
        f.write("%.6f\t%.6f\t%.6f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n" % \
                (t, hr, self.hrange, mbd_avg, mbd_std, pmax, \
                 np.std(p_l), calc_psr_phase(t), len(self.d)))
 
    def sum_to_file_old(self, f):

        nsumlst =   []
        mbdarr  =   []
        for nsum, e in self.d.items():

            nsumlst.append(nsum)
            mbdarr.append(e.mbd)

        mbdavg, mbdstd, nsumlst_sel   =   select_by_mbd(mbdarr, nsumlst)

        nsum_min    =   1024
        ph          =   0.0
        t           =   0.0
        hr_min      =   0.0

#        pharr   =   []
#        tarr    =   []
        parr    =   []
        for nsum, e in self.d.items():

            if not (nsum in nsumlst_sel):
                continue

            if nsum_min > nsum:
                nsum_min    =   nsum
                ph          =   e.phase
                t           =   e.time 
                hr_min      =   e.hrange

#            pharr.append(e.phase)
#            tarr.append(e.time) 
            parr.append(e.p)         
            mbdarr.append(e.mbd) 
         
        f.write("%.6f\t%.6f\t%.6f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n" % \
                (t, hr_min, self.hrange, mbdavg, mbdstd, np.average(parr), \
                 np.std(parr), ph, len(nsumlst_sel)))
        
    def __str__(self):
        return 'time: %f, hrange: %f, nitems: %d' % \
            (self.time, self.hrange, len(self.d))

    def insert_and_keep(self, ei):
#        if self.d.has_key(ei.nsum):
        if ei.nsum in self.d:
            self.d[ei.nsum].append(ei)
        else:
            self.d[ei.nsum] =   [ei]
        hr0  =   np.min([self.time - self.hrange, ei.time + ei.hrange])
        hr1  =   np.max([self.time + self.hrange, ei.time + ei.hrange])
        self.time   =   (hr0 + hr1) * 0.5
        self.hrange =   (hr1 - hr0) * 0.5

    def insert(self, ei):
        
        for nsum, e in self.d.items():
            if nsum != ei.nsum:
                if not is_overlap(e, ei):
                    return
        if self.d.has_key(ei.nsum):
            e   =   self.d[ei.nsum]
#            if  np.abs(e.mbd - ei.mbd) > 5.0: # if mbd diff > 5 ns, exclude
#                return
            e.time      =   (e.time + ei.time) * 0.5
            e.hrange    *=  0.5
            ei.time     =   e.time
            ei.hrange   =   e.hrange
            e.p         =   (e.p + ei.p) * 0.5
            e.phase     =   (e.phase + ei.phase) * 0.5
#            print 'nsum %d, time %f merged (%.3f|%.3f)' % (e.nsum, e.time, e.mbd, ei.mbd)
            e.mbd       =   merge_mbd(e.mbd, ei.mbd)
        else:
            self.d[ei.nsum]  =   ei

        if self.hrange < ei.hrange:
            self.time   =   ei.time
            self.hrange =   ei.hrange

class Match(object):
    def __init__(self):
        self.prefix     =   ''
#        self.nsum_list  =   []
#        self.aptime     =   0.0
#        self.sigma      =   0.0
        self.factor     =   0.0
        self.blid      =   -1

        self.cl         =   []

#        self.ne_min     =   0
        self.ds         =   {}

    def load_fitdump(self, dm, blid, nsum):

#        print 'nsum %d, noffset %d...' % (nsum, noffset)

        if self.ds == {}:
            self.ds =   np.load('No%04d/fitdump.npy' % (scan_no), allow_pickle = True).item()

#        print(self.ds)

        ps  =   self.ds[dm][blid][nsum]['p']
        t   =   self.ds[dm][blid][nsum]['t']
        mbd =   self.ds[dm][blid][nsum]['mbd']
        sbd =   self.ds[dm][blid][nsum]['sbd']
    
        ntot    =   len(ps)
        n0  =   0
        while n0 < ntot:
            if ps[n0] > 0.0:
                break
            n0  +=  1
        n1  =   ntot - 1
        while n1 >= 0:
            if ps[n1] > 0.0:
                break
            n1  -=  1

        if n1 - n0 <= 0:
            return []

        print('n0: %d, n1: %d' % (n0, n1))

        ps1 =   ps[n0:n1+1]
        ave =   np.average(ps1)
        std =   np.std(ps1)

        hrange  =   self.aptime * nsum * 0.5 * self.factor

        print ('nsum: %d, ave: %f, std: %f' % (nsum, ave, std))

        ps  =   (ps - ave) / std
        ids =   np.where(ps > self.sigma)[0]
     
        es  =   []
        for id in ids:
            e       =   Event()
            e.time  =   t[id]
            e.hrange=   hrange
            e.nsum  =   nsum
            e.p     =   ps[id]
            e.mbd   =   mbd[id]
            e.sbd   =   sbd[id]
            es.append(e)
        return es
    
    def insert_to_cl(self, e):
#        if len(self.cl) == 0:
#            self.cl.append(Candidate(e))
#            return
        has_overlap =   False
        for c in self.cl:
            if is_overlap(c, e):
                has_overlap =   True
                c.insert_and_keep(e)
                break
        if not has_overlap:
            self.cl.append(Candidate(e))

    def trim_cl(self):

        hambig  =   ambig * 0.5
#        hr  =   ambig / 4.0
        hr  =   2.0

        for i in range(len(self.cl)):
            
            c   =   self.cl[i]
            ne  =   len(c.d)
            if ne < self.ne_min:
                continue
# first loop to collect mbd and to calc avg, std
#            mbd_l   =   []
#            p_l     =   []
#            for nsum, e_l in c.d.items():
#                for e in e_l:
#                    mbd_l.append(e.mbd)
#                    p_l.append(e.p)
#            a   =   np.average(mbd_l, weights = p_l)
#            s   =   np.std(mbd_l)

# resolve ambig:
#            for k in range(len(p_l)):
#                if mbd_l[k] < a - hambig:
#                    mbd_l[k]    +=  ambig
#                if mbd_l[k] > a + hambig:
#                    mbd_l[k]    -=  ambig

# calculate average again:
#            a   =   np.average(mbd_l, weights = p_l)
            
# second loop to exclude mbd with large deviation
            dnew    =   OrderedDict()
            for nsum, e_l in c.d.items():
                l   =   []
                for e in e_l:
#                    if keep_mbd_ambig(e.mbd, a, hr):
                    if True:
                        l.append(e)
                if len(l) > 0:
                    self.cl[i].d[nsum]   =   l
                else:
                    self.cl[i].d.pop(nsum)
#            break

    def print_cl(self):

        fname   =   'No%04d/dm%.3f/bl%03d.log' % (scan_no, self.dm, self.blid)
        f       =   open(fname, 'w')
        for i in range(len(self.cl)):
            c   =   self.cl[i]
            ne  =   len(c.d)
#            if ne == len(self.nsum_list):
            if ne >= self.ne_min:
#                print '%d:' % (i)
                f.write('#%d:\n' % (i))
                for nsum, e_l in c.d.items():
#                    print nsum, e.time, e.hrange, e.p, e.phase
                    for e in e_l:
                        f.write(e.__str__() + '\n')
#                        print e
#                print ''
                f.write('###\n\n')
        f.close()

    def sum_cl(self):
        fname   =   'No%04d/dm%.3f/bl%03d.nsum' % (scan_no, self.dm, self.blid)
        f   =   open(fname, 'w')
        ncl =   len(self.cl)
        for i in range(ncl):
            c   =   self.cl[i]
            ne  =   len(c.d)
            if ne >= self.ne_min:
                c.sum_to_file_max_power(f)  
        f.close()

def match(cfg, dm, blid):

    m   =   Match()
#    m.bl_no =   259
    m.blid  =   blid
    m.dm    =   dm

    m.nsum_list =   cfg.nsums

#    m.nsum_list =   [4, 8, 16, 32]
#    m.ne_min    =   2
#    m.sigma   =   5.0

#    m.nsum_list =   [2, 4, 8, 16]
    m.ne_min    =   cfg.ne_min_winmatch
    m.sigma     =   cfg.sigma_winmatch
    
    m.nsum_list =   np.sort(m.nsum_list)[::-1]

    m.prefix  =   '.'
    m.aptime  =   cfg.ap
    m.factor  =   1.0

    for nsum in m.nsum_list:

        es  =   m.load_fitdump(dm, m.blid, nsum)

        t_arr   =   []
        for e in es:
            t_arr.append(e.time) 
        ids =   np.argsort(t_arr)
        for id in ids:
            m.insert_to_cl(es[id])

#    m.trim_cl()
    m.print_cl()
    m.sum_cl()

if __name__ == '__main__':
    global scan_no
    scan_no =   int(sys.argv[1])
    cfg =   utils.gen_cfg(scan_no)
#    with open('No%04d/blinfo.txt' % (scan_no), 'r') as f:
#        nbl =   len(f.readlines())

    for dm in cfg.dms:
        path    =   'No%04d/dm%.3f' % (scan_no, dm)
        if not os.path.exists(path):
            os.mkdir(path)
        for blid in range(cfg.nbl):
            match(cfg, dm, blid)
    
