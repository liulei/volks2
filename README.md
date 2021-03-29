# VOLKS2
### VLBI Observation single pulse Localization Keen Searcher (2nd release)

## Introduction
This is the second release of the VOLKS pipeline. Significant updates have been made since the last release. 

This pipeline is designed to conduct single pulse search and localization in regular VLBI observations. Unlike the radio imaging based pipeline, in VOLKS, the search and localization are two independent steps. The search step takes the idea of geodetic VLBI post processing, which fully uses the cross spectrum fringe phase information to maximize the signal power. Compared with auto spectrum based method, it is able to extract single pulses from highly RFI contaminated data. The localization uses the astrometric solving methods, which derives the single pulse location by solving a set of linear equations given the relation between the residual delay and the offset to a priori position. 

## Main features

- Flexible configurations: arbitrary selection of IFs, polarizations and baselines; easy adjustment of search parameters;
- Single pulse search, multiple baselines match and localization with no extra software dependent.  
- Full parellelization (with `mpi4py`) and GPU support (with `PyTorch` and `CuPy`). Optimized for multiple nodes GPU clusters.

The whole pipeline has been extensively tested with the data of EVN observation `el060`. 

## Acknowledgement
The development of this pipeline is supported by the National Science Fundation of China (No. 11903067).  

**Note**: If you make use of VOLKS pipeline in your publication, we require that you quote the pipeline web address https://github.com/liulei/volks2 and cite the following papers:

- `Liu, L., Tong, F., Zheng, W., Zhang, J. & Tong, L. 2018, AJ, 155, 98`, which describes the non-imaging single pulse search method.
- `Liu, L., Zheng, W., Yan, Z. & Zhang, J. 2018, Research in Astronomy and Astrophysics, 18, 069`, which compares the cross spectrum based method and the auto based spectrum method for single pulse serach in VLBI observation.
- `Liu, L., Jiang, W., Zheng, W., et al. 2019, AJ, 157, 138`, which describes the radio imaging and astrometric solving single pulse localzation methods.

Please do not hesitate to contact me (liulei@shao.ac.cn) if you have any quation.

## Requirement

- Linux or MacOS system.
- gcc, gfortran, Python3, numpy, ctypes, matplotlib (for making )

## Compile
```
cd calc9.1
make
```
This will generate `libcalc_cwrapper.so`. The modification is based on `calcserver` distributed with DiFX-2.4:

- Add variable `PUTDSTRP`, `PARTIAL` to `cdrvr.f`, `calcmodl2.f` and `cpart.i`, so as to obtain partial derivatives of delay with Ra and Dec. 
- Provide C wrapper for `CALC` (`libcalc_cwrapper.so`), which will be called by `solve_all.py` with `ctypes` to obtain partial derivatives.

## Run

The author has tried his best to make the pipeline easy to understand and use. However, due to the complexity of VLBI data processing, it still requires some effort to have it run and give the final result. Since the Python code is self-explanatory, it is stronly suggested that the user read the source code and figure out how it works. I will give short explanation (**Description**, **Input** and **Output**) for each step.

### Prepare configuration: `utils.py`
**Description**:

- All programs in the pipeline will first call `utils.gen_cfg()` to get the configuration, so as to avoid modifications to those programs. Therefore one need to set specific task in `gen_cfg()` and then prepare the corresponding configuration file, e.g. `gen_cfg_el060()`.


### Setup environment
- Run command `source environment`.
- This will tell `calc` where to find `JPLEPH` and `Horizons.lis`. Please keep other settings unchanged.



### Calibration: `genswincal.py`

**Description**:

- Carry out fringe fitting for calibrtion source. PCAL and channel delay are set to zero and are output after fitting.

**Input**: 

- DiFX calibration scan, baseline, frequency information.
- Fill in required info for `DiFX()`, `DiFXScan()` and `DataDescriptor()`. **You have to specify baseline and frequency information manually!** 
- Specify fitting details in `fit_multiband()`, e.g, FFT size for MBD (`nmb`) and SBD (`nsb`) search. The PCAL and channel delay fitting results will be print out.

**Output**: 

- PCAL and channel delay for each frequency channel (IF in AIPS) are print out.

### Fringe fitting: `genswindump.py`

**Description:**

- Carry out fringe fitting for each fast dumped visibilities (time segment in Liu et al. 2018a) of several given re-sampling time (`nsum`).

**Input**: 

- Fill in `pcal_dict{}` and `sbd_dict{}` with PCAL and channel delay information derived in previous step. 
- Configurations of `DifX()` and `DataDescriptor()` are similar with `genswincal.py`. 
- Specify re-sampling time in `nsum_list`. `nsum` means number of accumulation period to be summed. 
- Specify fitting details in `fit_multiband()`, e.g, FFT size for MBD (`nmb`) and SBD (`nsb`) search. 

**Output**: 

- `blxxx_sumxxx_offsetxxx.fitdump`, records time, width and pulsar phase of each time segment in the given re-sampling time.

### Windows filtering: `winmatch.py`

**Description**:

- First pick up single pulses from fast dumped data of each re-sampling time according to given threshold, then counting how many windows (re-sampling time) in which they are detected. 
- Single pulses are output if they are detected in at least `ne_min` windows.

**Input**:

- `blxxx_sumxxx_offset.fitdump` files in previous step.
- Specify re-sampling time in `nsum_list`, which should be identical with `genswindump.py`. 
- Specify filtering parameters, including threshold to pick up single pulses (`sigma`), minimum number of detected windows (`ne_min`).

**Output**:

- `blxxx.nsum`, records time, width and pulsar phase of each single pulse candidate after multiple window size filtering.

### Multiple baselines cross matching: `crossmatch.py`

**Description**:

- Cross matching single pulses detected on multiple baselines.
- A single pulse is output if it is detected on at least `count_min` baselines.

**Input**:

- 'blxxx.nsum' files in previous step.
- Specify `count_min`.

**Output**:

- 'scanxxx_sss.ssssss.sp', generated for every individual single pulses. Records baseline id it is detected, time, width and pulsar phase on this baseline.

### Combine all `.sp` files: `combsp.sh`

**Description**:

- Run command `./combsp.py > xx.sp`, `xx` means scan number.

**Input**:

- `scanxxx_sss.ssssss.sp`

**Output**:

- `xx.sp`, `xx` means scan number.

### Extract single pulse from swin files: `extractswin.py`

**Description**:

- Extract records specified in `xx.sp` from original SWIN files.

**Input**:

- `xx.sp` files in previous step. 
- Original SWIN files of pulsar scans:.

**Output**:

- SWIN files that only contain records specified in `xx.sp`. 

### Generate FITS-IDI files `difx2fits`

**Description**:

- Run `difx2fits <SWIN files>` to convert SWIN files to FITS-IDI format.
- The `difx2fits` program is distributed together with DiFX. We use 2.4 version. 
- Note! DiFX-2.4 could not resolve accumulation periods (APs) as short as 1 ms appropriately in `difx2fits`. Our hacking is in `fitUV.c`, reducing the threshold in line 745 from 1.0/86400000 further to 0.01/86400000. Besides that, this program only works by removing `-O2` optimization when compiling.

**Input**:

- SWIN files that only contain single pulse records generated in previous step.
- SWIN files of calibration and phase reference scans.

**Output**:

- FITS-IDI file that contains single pulse visibilities.
- FITS-IDI files of calibation and phase reference scans.

### Phase reference calibration: `AIPS`

**Description**:

- In this pipeline, localization is independent of single pulse search and is not time consuming. At present, we use AIPS to calibrate the extracted single pulse visibilities. We are planning to integrate this part into the pipeline, so as to make the whole data processing more convenient.
- In task `SPLIT`, `aparm(0)` is set to 0, such that channels inside one IF are not avaraged. This keeps the fringe phase ambiguity information, which makes it possible to localize burst far from phase center.

**Input**:

- FITS-IDI files of single pulses, calibration and phase reference source.

**Output**:

- FITS-IDI file of single pulses after calibration, ready to localize. 

### Localization: `solve_all.py`

**Description**:

- Localize single pulses using both imaging and astrometric solving method. 
- Visibilities in FITS-IDI file that belong to one single pulse are first grouped together. Then each of these single pulses are localized with `solve()` and `imaging()` independently.

**Input**:

- FITS-IDI file of single pulses after calibration.  
- EOP.txt which contains the eop information during observation.
- For radio imaging, specify cell size and number of grids in each dimension.

**Output**:

- `loc_sp.npy`, a 2D floating type array. Each row corresponds to a single pulse. The meaning of each column is:
    - col 0: baseline mask. For baseline `id` (0 indexed), `baseline_mask & (1 << id)` is 1 if single pulse is detected. Note `baseline_mask` is `float` type when it is retrieved from the array. You have to convert it to `int` type before using.
    - col 1: Ra by solving (in mas)
    - col 2: Ra uncertainty by solving (in mas)
    - col 3: Dec by solving (in mas)
    - col 4: Dec uncertainty by solving (in mas)
    - col 5: time (in mjd)
    - col 6: Ra by imaging (in mas)
    - col 7: Dec by imaging (in mas)
    - col 8: beam semi-minor axis (in mas)
    - col 9: beam semi-major axis (in mas)
    - col 10: major axis position angle (in degree)
