#!/usr/bin/env python

import sys
import numpy as np
import ctypes as c
import ctypes
from collections import OrderedDict
import copy

class CIn(c.Structure):

    _fields_    =   [   ('date', c.c_int), \
                        ('neop', c.c_int), \
                        ('time', c.c_double), \
                        ('ra', c.c_double), \
                        ('dec', c.c_double), \
                        ('x', c.c_double), \
                        ('y', c.c_double), \
                        ('z', c.c_double), \
                        ('axis_off', c.c_double), \
                        ('EOP_time', c.POINTER(c.c_double)), \
                        ('tai_utc', c.POINTER(c.c_double)), \
                        ('ut1_utc', c.POINTER(c.c_double)), \
                        ('xpole', c.POINTER(c.c_double)), \
                        ('ypole', c.POINTER(c.c_double)), \
                        ('stnname', c.c_char_p), \
                        ('srcname', c.c_char_p)]

#def set_site(din, s):
#    din.x   =   s['x']
#    din.y   =   s['y']
#    din.z   =   s['z']
#    din.axis_off    =   s['axis_off']
#    din.stnname     =   s['stnname']
#
#def set_source(din, s):
#    din.srcname =   s['srcname']
#    din.ra      =   s['ra']
#    din.dec     =   s['dec']
#
#def set_datetime(din, dt):
#    din.date    =   dt['date']
#    din.time    =   dt['time']

def set_eop(din, name_vex):

    eops    =   load_eop(name_vex) 
    neop    =   len(eops)
    
    EOP_time    =   np.zeros(neop, dtype = float) 
    tai_utc     =   np.zeros(neop, dtype = float) 
    ut1_utc     =   np.zeros(neop, dtype = float) 
    xpole       =   np.zeros(neop, dtype = float) 
    ypole       =   np.zeros(neop, dtype = float) 
    
    for i, eop in enumerate(eops):
        EOP_time[i] =   eop['eop_ref_epoch']
        tai_utc[i]  =   eop['TAI-UTC']
        ut1_utc[i]  =   eop['ut1-utc']
        xpole[i]    =   eop['x_wobble']
        ypole[i]    =   eop['y_wobble']

    din.neop        =   neop
    din.EOP_time    =   EOP_time.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    din.tai_utc     =   tai_utc.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    din.ut1_utc     =   ut1_utc.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    din.xpole       =   xpole.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    din.ypole       =   ypole.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

class COut(c.Structure):

    _fields_    =   [   ('delay',   c.c_double), \
                        ('rate',    c.c_double), \
                        ('dry',     c.c_double), \
                        ('wet',     c.c_double), \
                        ('u',       c.c_double), \
                        ('v',       c.c_double), \
                        ('w',       c.c_double), \
                        ('pd_pra',  c.c_double), \
                        ('pd_pdec', c.c_double)]

# doy: 1 ~ 366
def doy2mjd(year, doy):
    return doy - 678576 + 365 * (year - 1) + \
            (year - 1) / 4 - (year-1)/100+(year-1)/400

def get_mjd(s):
    s       =   s.strip()
    s1      =   s.split('y')
    year    =   int(s1[0])
    s2      =   s1[1].split('d')
    doy     =   int(s2[0])
    return doy2mjd(year, doy)

def load_eop(eop_filename):
    
    try:
        f    =    open(eop_filename, 'r')
        lines    =    f.readlines()
        f.close()
    except IOError:
        print ('Cannot findfile %s to read!' % (eop_filename))
        sys.exit(0)
    
    find_eop    =    False
    for id0, line in enumerate(lines):
        if '$EOP;' in line:
            find_eop    =    True
            break

    if not find_eop:
        print ('Cannot find any valid eop record in %s!' % (eop_filename))
        sys.exit(0)

    d       =   OrderedDict()
    eops    =    []
    for i in range(id0+1, len(lines)):
        line    =    lines[i].rstrip()
        if 'def ' in line:
            d    =    OrderedDict()
        if '=' in line:
            l    =    line.split('=')
            key    =    l[0].strip()
            val    =    l[1].split()[0]
            if key == 'eop_ref_epoch':
                d[key]  =   float(get_mjd(val))
            elif 'TAI-UTC' in key or 'ut1-utc' in key or 'wobble' in key:
                d[key]  =   float(val)
        if 'enddef' in line:
            if len(d) > 0:
                eops.append(d)
                d   =   OrderedDict()

    print('Total eops in file: %d' % (len(eops)))
#    for i, eop in enumerate(eops):
#        print('eop %d: ' % (i+1))
#        for k, v in eop.iteritems():
#            print k, v
#        print ''

    return eops
