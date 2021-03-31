#!/usr/bin/env python

import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.lines as mlines
#import matplotlib.patches as patches
import sys
import utils

dtype_cand  =   np.dtype([  ('t',       'f8'), \
                            ('hr_min',  'f8'), \
                            ('hr_max',  'f8'), \
                            ('mbdavg',  'f8'), \
                            ('sbd',     'f8'), \
                            ('pavg',    'f8'), \
                            ('pstd',    'f8'), \
                            ('psrph',   'f8'), \
                            ('nsum',    'i4')])

scan_no     =   -1
#ph0_dict    =   {69: 0.581375, 71: 0.414097, 73: 0.246818}

def calc_psr_phase(t):

    c1  =   -6.89262836039208383e-03
    F0  =   1.399538059693
    return t * (F0 + c1 / 60.)
#    ph0 =   0.581375 # scan 69
#    ph0 =   0.414097 # scan 71
#    ph0 =   0.246818 # scan 73
#    ph0 =   ph0_dict[scan_no]
#    return (t - 10.000512) * (F0 + c1 / 60.) + ph0

def is_overlap(cand, cm):
    if np.abs(cand.t - cm.t) <= cand.hr + cm.hr:
        return True
    return False

def is_psr(t, hr):

    F0  =   1.399538059693
    dph =   F0 * hr
    
    ph  =   calc_psr_phase(t)
    ph  -=  np.floor(ph)
    if np.abs(ph - 0.978) <= 0.005 + dph:
        return True
    return False

class Candidate(object):
    
    def __init__(self, a, blid):
        self.t  =   a['t']
        self.hr =   a['hr_min']
#        self.hr =   a['hr_max']
        self.mbd =   a['mbdavg']
        self.p   =   a['pavg']
        self.pstd   =   a['pstd']
        self.psrph  =   a['psrph']
        self.nsum   =   a['nsum']
        self.blid   =   blid
        
class CrossMatch(object):

    def __init__(self, cand):
        
        self.t      =   cand.t
        self.count  =   1
        self.hr     =   cand.hr
        self.d      =   {}
        self.d[cand.blid]=   cand
        self.psrph  =   [cand.psrph]
        self.bls    =   [cand.blid]

    def insert_cand(self, cand):
        
#        if self.d.has_key(cand.blid):
        if cand.blid in self.d:
            print('Warning! bl %d, t %.6f already inserted (t %.6f)!' % \
                    (cand.blid, cand.t, self.t))
            if cand.p > self.d[cand.blid].p:
                self.d[cand.blid] = cand
                t_min   =   np.min([self.t - self.hr, cand.t - cand.hr])
                t_max   =   np.max([self.t + self.hr, cand.t + cand.hr])
                self.t  =   (t_min + t_max) * 0.5
                self.hr =   (t_max - t_min) * 0.5
                return

        self.count  +=  1
        t_min   =   np.min([self.t - self.hr, cand.t - cand.hr])
        t_max   =   np.max([self.t + self.hr, cand.t + cand.hr])
        self.t  =   (t_min + t_max) * 0.5
        self.hr =   (t_max - t_min) * 0.5
        self.d[cand.blid]   =   cand
        self.psrph.append(cand.psrph)
        self.bls.append(cand.blid)

def load_cand_file(dm, blid):

    fname   =   'No%04d/dm%.3f/bl%03d.nsum' % (scan_no, dm, blid)
    print('load %s...' % (fname))
    arr     =   np.loadtxt(fname, dtype = dtype_cand, ndmin = 1) 
    arr['mbdavg']  /=  1E9
    arr['sbd']  /=  1E9
    if len(arr) == 0:
        return []
    cands   =   []
    for a in arr:
        cands.append(Candidate(a, blid)) 
    return cands

def match_cand(cands, cms):
    
#    npsr    =   0
    for cand in cands:
#        if is_psr(cand.t, cand.hr):
#            npsr    +=  1
        inserted    =   False
        for cm in cms:
            if is_overlap(cand, cm):
                cm.insert_cand(cand) 
                inserted    =   True
                break
        if not inserted:
            cms.append(CrossMatch(cand))
#    print 'npsr from match_cand(): %d' % (npsr)

def is_indep_bl(cfg, cm):
    
    ncand   =   len(cm.d)
    ds    =   {}
    for blid, cand in cm.d.items():
        s1, s2  =   cfg.blid2name[blid] 
        if s1 in ds:
            ds[s1]  +=  1
        else:
            ds[s1]  =   1

        if s2 in ds:
            ds[s2]  +=  1
        else:
            ds[s2]  =   1

    count_max   =   np.max(list(ds.values()))
    if count_max < ncand:
        return True
    
    for s, c in ds.items():
        if c == count_max:
            print('All baselines are related with %s (%d bls in total)! skip.' % (s, c))
         
    return False

def output_cms(cfg, dm, cms):

    nbl_min =   cfg.nbl_min_crossmatch

    ncm     =   np.zeros(cfg.nbl, dtype = float)
    for cm in cms:

        if cm.count < nbl_min:
            continue

        if not is_indep_bl(cfg, cm):
            continue
 
        for blid, cand in cm.d.items():
            print('dm: %.3f, blid: %d, SNR: %.3f, mbd: %.3f' % \
                (dm, blid, cand.p, cand.mbd*1E9))

        print('')
        ncm[cm.count - 1]     +=  1

        f   =   open('No%04d_dm%.3f_%.6f.sp' % (scan_no, dm, cm.t), 'w')
        f.write('#blid\tapid\tnap\ttime\twidth\tmbd\tpower\n')
        for blid, cand in cm.d.items():
            i   =   int((cand.t - cand.hr - 0.0) / 1.024E-3 + 0.1)
            w   =   int(cand.hr / 1.024E-3 + 0.1) * 2
            bl_no   =   cfg.bl_nos[blid]
            name    =   cfg.bl_no2name[bl_no]
            f.write('%d\t%d\t%d\t%.6f\t%.6f\t%12.3f\t%.3f\t# %s\n' \
                    % (blid, i, w, cand.t, \
                        cand.hr * 2, cand.mbd*1E9, cand.p, name))

        f.close()
    print('ncm: ', ncm)

def main():

    if len(sys.argv) < 2:
        print('../crossmatch.py scan_no')
        sys.exit(0)

    global scan_no, nbl
    scan_no =   int(sys.argv[1])

    cfg =   utils.gen_cfg_el060(scan_no)
#    cfg =   utils.gen_cfg_aov025(scan_no)

    for dm in cfg.dms:
        cms =   []
        for blid in range(cfg.nbl):
            cands   =   load_cand_file(dm, blid) 
            if len(cands) == 0:
                continue
            match_cand(cands, cms) 

        output_cms(cfg, dm, cms)

#    plot_cms(cms)

if __name__ == '__main__':
    main()
